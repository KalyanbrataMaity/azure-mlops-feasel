import argparse
from pathlib import Path
import os
import pandas as pd


parser = argparse.ArgumentParser("prep")
parser.add_argument("--input_data", type=str, help="Name of the folder containing input data for this operation")
parser.add_argument("--output_data", type=str, help="Name of the folder we will write results out to")

args = parser.parse_args()

print("Replace missing values...")

lines = [
    f"Input data path: {args.input_data}",
    f"Output data path: {args.output_data}",
]

for line in lines:
    print(line)

print(os.listdir(args.input_data))

file_list = []
for filename in os.listdir(args.input_data):
    print("Reading file: %s..." % filename)
    with open(os.path.join(args.input_data, filename), "r") as f:
        input_df = pd.read_csv(Path(args.input_data) / filename)
        file_list.append(input_df)

# concatenate the list of Python dataframes
df = pd.concat(file_list)
#df = df.head(10000)


# we need to find and replace missing values for police district
# If a value is missing, replace it with 0
print("Replacing missing police districts...")
df["Police_District"] = df["Police_District"].fillna(0, inplace=True)

# write the results out for the next step
print("Writing results to output folder...")
df.to_csv((Path(args.output_data) / "ReplacedMissingFeatures.csv"), index=False)
print("Done replacing missing values.")