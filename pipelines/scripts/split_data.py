import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser("prep")
parser.add_argument("--input_data", type=str, help="Name of the folder containing input data for this operation")
parser.add_argument("--output_data_train", type=str, help="Name of the folder we will write training results out to")
parser.add_argument("--output_data_test", type=str, help="Name of the folder we will write testing results out to")

args = parser.parse_args()

print("Performing feature selection...")

lines = [
    f"Input data folder: {args.input_data}",
    f"Output train data folder: {args.output_data_train}",
    f"Output test data folder: {args.output_data_test}",    
]

for line in lines:
    print(line)

print(os.listdir(args.input_data))

file_list = []
for filename in os.listdir(args.input_data):
    print(f"Reading file: %s ..." % {filename})
    with open(os.path.join(args.input_data, filename), "r") as f:
        input_df = pd.read_csv((Path(args.input_data) / filename))
        file_list.append(input_df)

# concatenate the list of dataframes
df = pd.concat(file_list)

# We will have a relatively smaller test dataset size
# but this is still ~200K rows and we're going to use some
# of the data for hyperparameter sweeps
train_df, test_df = train_test_split(df, test_size=0.2, random_state=11084)

# Write the results for the next step
print("Writing results out ...")
train_df.to_csv((Path(args.output_data_train) / "TrainData.csv"), index=False)
test_df.to_csv((Path(args.output_data_test) / "TestData.csv"), index=False)

print("Done with feature selection step.")

