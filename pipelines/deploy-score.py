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
from azure.ai.ml.entities import (
    BatchEndpoint,
    ModelBatchDeployment,
    ModelBatchDeploymentSettings,
    Model,
    AmlCompute,
    Data,
    BatchRetrySettings,
    CodeConfiguration,
    Environment,
)
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential, AzureCliCredential
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component
from azure.ai.ml.constants import AssetTypes, BatchDeploymentOutputAction

from azure.ai.ml.sweep import (
    Choice,
    Uniform,
)


# NOTE: set your workspacename here!
workspace_name = "CS_ML_FEASELKL_WS"

# NOTE: if you do not have the cpu-cluster already, we will create one
# Alternatively, change the name to a CPU-based compute cluster 
cluster_name = "cpu-cluster"

# for local runs, I'm using Azure CLI credential
# for production runs as part of an MLOps configuration using Azure DevOps or Github Actions,
# recommend to use DefaultAzureCredential
# ml_client = MLClient.from_config(DefaultAzureCredential())  # for production
ml_client = MLClient.from_config(AzureCliCredential())  # for local dev
ws = ml_client.workspaces.get(workspace_name)

# make sure the compute cluster exists already
try:
    cpu_cluster = ml_client.compute.get(cluster_name)
    print(f"You already have a cluster named {cluster_name}, we'll reuse as it is.")
except Exception:
    print("Creating a new cpu compute target...")

    # let's create the azure machine learning compute object with the intended parameters
    # if you run into an out of quota error, change the size to a comparable VM that is 
    # available in your region
    cpu_cluster = AmlCompute(
        name=cluster_name,
        type="amlcompute",  # aml is the on-demand VM service
        size="Standard_DS3_v2",  # VM family 
        min_instances=1,  # minimum running nodes when there are no jobs running
        max_instances=2,  # maximum number of nodes in cluster
        idle_time_before_scale_down=180,  # node running after the job termination
        tier="Dedicated",  # dedicated or low-priority, the latter is cheaper but there is a chance of job termination
    )
    print(f"AMLCompute with name {cpu_cluster.name} will be created, with compute size {cpu_cluster.size}")

    # now we pass the object to MLClient's create_or_update method
    cpu_cluster = ml_client.compute.begin_create_or_update(cpu_cluster)

# Ensure that there is an endpoint for batch scoring
endpoint_name = "chicago-parking-tickets-endpoint"
try:
    endpoint = ml_client.batch_endpoints.get(endpoint_name)
    print(f"Batch endpoint {endpoint_name} already exists.")
except Exception:
    print(f"Creating a new batch endpoint {endpoint_name}...")
    endpoint = BatchEndpoint(
        name=endpoint_name,
        description="Batch scoring endpoint for Chicago Parking Tickets payments status",
    )
    ml_client.batch_endpoints.begin_create_or_update(endpoint).result()
    endpoint = ml_client.batch_endpoints.get(endpoint_name)
    print(f"Batch endpoint {endpoint.name} created successfully.")

# Retrieve the parking tickets model
model = ml_client.models.get(
    name="ChicagoParkingTicketsCodeFirst",
    version="2",
)
print("Retrieved model.")

# Get the correct environment
deployment_name = "cpt-batch-deployment"
try:
    deployment = ml_client.batch_deployments.get(
        endpoint_name=endpoint_name, name=deployment_name
    )
    print(f"Batch deployment {deployment_name} already exists.")
except Exception:
    print(f"No deployment exists, creating a new deployment...")
    
    # Define the environment for the deployment
    deployment_env = ml_client.environments.get(name="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu", version='33') 
    deployment = ModelBatchDeployment(
        name=deployment_name,
        description="Batch scoring of Chicago Parking Tickets payments status",
        endpoint_name=endpoint.name,
        model=model,
        environment=deployment_env,
        code_configuration=CodeConfiguration(
            code="scripts",
            scoring_script="score_model.py",
        ),
        compute=cluster_name,
        settings=ModelBatchDeploymentSettings(
            instance_count=1,
            max_concurrency_per_instance=2,
            mini_batch_size=10,
            output_action=BatchDeploymentOutputAction.APPEND_ROW,
            retry_settings=BatchRetrySettings(max_retries=3, timeout=600),
            logging_level="info",
        ),
    )
    ml_client.batch_deployments.begin_create_or_update(deployment).result()
    print("Created deployment.")

    # Now make the deployment the default for our endpoint
    endpoint.defaults.deployment_name = deployment.name
    ml_client.batch_endpoints.begin_create_or_update(endpoint).result()
    print('Made deployment the default for this endpoint.')


# Prepare the dataset
data_path = "data/score_data"
dataset_name = "ChicagoParkingTicketsUnlabeled"

# verify data path exists locally
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data path {data_path} doesnot exists. Please ensure scoring data is present.")

try:
    # try to get existig dataset
    chicago_dataset_unlabeled = ml_client.data.get(
        name=dataset_name, label="latest"
    )
    print(f"Dataset {dataset_name} already exists.")

    # verify dataset path exists
    if not chicago_dataset_unlabeled.path:
        raise ValueError(f"Dataset path is empty")
    print(f"Dataset path verified: {chicago_dataset_unlabeled.path}")
    
except Exception as e:
    print(f"No dataset exists, creating a new dataset due to {e}")
    chicago_dataset_unlabeled = Data(
        path=data_path,
        type=AssetTypes.URI_FOLDER,
        description="An unlabelled dataset for Chicago Parking Ticket payment status",
        name=dataset_name,
        version="1.0.0"
    )
    ml_client.data.create_or_update(chicago_dataset_unlabeled)
    chicago_dataset_unlabeled = ml_client.data.get(
        name=dataset_name, label="latest"
    )
    print(f"Dataset {chicago_dataset_unlabeled.name} created successfully.")

# NOTE: If youa re getting a "BY POLICY" error, make sure that your account is an Oewner
# on the Azure ML workspace. You must *explicitely* grant rights, even if you are the subscription owner!

# Create a job to run the batch scoring
# Add error handling for batch job
try:
    print(f"Starting batch scoring job using endpoint {endpoint.name}")
    batch_job = ml_client.batch_endpoints.invoke(
        endpoint_name=endpoint.name,
        input=Input(type=AssetTypes.URI_FOLDER, path=chicago_dataset_unlabeled.path)
    )
    print(f"Batch scoring job {batch_job.name} started successfully")
except Exception as e:
    print(f"Error invoking batch endpoint: {str(e)}")
    raise

# wait for the job to finish and then access the data via the AML interface.
# Jobs > chicogo-parking-tickets-batch > {job name} > BatchScoring > Outputs + logs > data outputs (show data outputs!)