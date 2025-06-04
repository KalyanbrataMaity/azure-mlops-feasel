import sys
import os
import timeit
from datetime import datetime
import numpy as np
import pandas as pd
from random import randrange
import urllib
from urllib.parse import urlencode

import azure.ai.ml
from azure.ai.ml import MLClient, Input, Output
from azure.ai.ml.entities import Workspace, AmlCompute, Data
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential, AzureCliCredential
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component
from azure.ai.ml.constants import AssetTypes



# set your workspacename here
workspace_name = "CS_ML_FEASELKL_WS"

# if we do not have the cpu-cluster already, we willc reate it
cluster_name = "cpu-cluster"

# for loacl rus, Im using Azure CLI credential 
# for production runs as part of an MLOps configuration using Azure DevOps or Github Actions,
# recommend to use DefaultAzureCredential

#ml_client = MLClient.from_config(DefaultAzureCredential()) # for production
ml_client = MLClient.from_config(AzureCliCredential()) # for local dev
ws = ml_client.workspaces.get(workspace_name)

# make sure the compute cluster exists already
try:
    cpu_cluster = ml_client.compute.get(cluster_name)
    print(f"You already have a cluster named {cluster_name}, we'll reuse as it is.")

except Exception:
    print("Creating a new cpu compute target...")

    # let's create the azure machine learning copute object with the intended parameters
    # if you run into an out of quota error, change the size to a comparable VM that is 
    # available in your region
    cpu_cluster = AmlCompute( 
        name=cluster_name,
        type="amlcompute", # aml is the on-demand VM service
        size="Standard_DS3_v2", # VM family 
        min_instances=0, # minimum running nodes when there are no jobs running
        max_instances=4, # maximum number of nodes in cluster
        idle_time_before_scale_down=180, # node running after the job termination
        tier="Dedicated", # dedicated or low-priority, the latter is cheaper but there is a chance of job termination
    )
    print(f"AMLCompute with name {cpu_cluster.name} will be created, with compute size {cpu_cluster.size}")

    # now we pass the object to MLClient's create_or_update method
    cpu_cluster = ml_client.compute.begin_create_or_update(cpu_cluster)

## upload training data
print("Uploading training data to Azure ML Workspace ...")
train_data_path = "data/train_data"

# Verify training data path exists
if not os.path.exists(train_data_path):
    raise FileNotFoundError(f"Training data path {train_data_path} does not exist.")

try:
    # check if dataset already exists
    training_dataset = ml_client.data.get(
        name="ChicagoParkingTicketsFolder",
        version="1.0.0"
    )
    print(f"Training dataset already exists at: {training_dataset.path}")
except Exception:
    print("Creating new training dataset...")
    training_dataset = Data(
        path=train_data_path,
        type=AssetTypes.URI_FOLDER,
        description="Training dataset for chicago parking tickets model",
        name="ChicagoParkingTicketsFolder",
        version="1.0.0"
    )
    ml_client.data.create_or_update(training_dataset)
    training_dataset = ml_client.data.get(
        name="ChicagoParkingTicketsFolder", label="latest"
    )
    print(f"Dataset {training_dataset.name} created successfully.")

    # verify dataset is accessible
    print(f"Verifying dataset accessibility at: {training_dataset.path}")

parent_dir = "./config"

# Performing data preparation
# all the feature-* and split-data steps can be combined together
# but have left them separate to show an example of multi-step pipelines

replace_missing_values = load_component(source=os.path.join(parent_dir, "feature-replace-missing-values.yml"))
feature_engineering = load_component(source=os.path.join(parent_dir, "feature-engineering.yml"))
feature_selection = load_component(source=os.path.join(parent_dir, "feature-selection.yml"))
split_data = load_component(source=os.path.join(parent_dir, "split-data.yml"))
train_model = load_component(source=os.path.join(parent_dir, "train-model.yml"))
register_model = load_component(source=os.path.join(parent_dir, "register-model.yml"))


@pipeline(name="training_pipeline", description="Build a Training pipeline")
def build_pipeline(raw_data):
    step_replace_missing_values = replace_missing_values(input_data=raw_data)
    step_feature_engineering = feature_engineering(input_data=step_replace_missing_values.outputs.output_data)
    step_feature_selection = feature_selection(input_data=step_feature_engineering.outputs.output_data)
    step_split_data = split_data(input_data=step_feature_selection.outputs.output_data)

    train_model_data = train_model(train_data=step_split_data.outputs.output_data_train,
                                   test_data=step_split_data.outputs.output_data_test,
                                   max_leaf_nodes=128,
                                   min_samples_leaf=32,
                                   max_depth=12,
                                   learning_rate=0.1,
                                   n_estimators=100)
    register_model(model=train_model_data.outputs.model_output, test_report=train_model_data.outputs.test_report)
    return {"model": train_model_data.outputs.model_output,
            "report": train_model_data.outputs.test_report}

def prepare_pipeline_job(cluster_name: str):
    # !!! must have a dataset already in place
    cpt_asset = ml_client.data.get(name="ChicagoParkingTicketsFolder", version="1.0.0")
    raw_data = Input(type='uri_folder', path=cpt_asset.path)
    pipeline_job = build_pipeline(raw_data)

    # set pipeline level compute
    pipeline_job.settings.default_compute = cluster_name

    # set pipeline level datastore
    pipeline_job.settings.default_datastore = "workspaceblobstore"
    pipeline_job.settings.force_rerun = True
    pipeline_job.display_name = "train_pipeline"
    return pipeline_job


prepped_job = prepare_pipeline_job(cluster_name)
ml_client.jobs.create_or_update(prepped_job, experiment_name="Chicago Parking Tickets Code-First")
print("Now look in the Azure ML Jobs UI to see the status of the pipeline job. This will be in the 'Chicago Parking Tickets Code-First' experiment.")

