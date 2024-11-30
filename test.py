#%%
import pandas as pd
import os

#%% Adjust display settings
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)

#%% Set the directory to the project root (current directory)
directory = os.getcwd()

#%% Get all CSV files in the project root directory
csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]

#%% Load the datasets from the project root directory
data_frames = {}
for file in csv_files:
    file_path = os.path.join(directory, file)
    name = os.path.splitext(file)[0]  # Use the filename without extension as the key
    data_frames[name] = pd.read_csv(file_path)

#%% Manually load the CSV file from the subdirectory
subdirectory_file = os.path.join(directory, 'Crop Yield Prediction Dataset', 'archive', 'yield_df.csv')
data_frames['yield_df'] = pd.read_csv(subdirectory_file)

#%% Data set structure and columns
for name, df in data_frames.items():
    print(f"Dataset Name: {name}")
    print(df.head())  # Print the first 5 rows
    print()

#%% Specify the output file name
output_file = "dataset_info.txt"
with open(output_file, "w") as file:
    for name, df in data_frames.items():
        # Write the dataset name
        file.write(f"Dataset Name: {name}\n")

        # Write the first 7 rows
        file.write(df.head(5).to_string(index=False))
        file.write("\n\n")

        # Write the columns
        file.write(f"Columns for {name}: {df.columns.tolist()}\n")
        file.write("-" * 50 + "\n")  # Separator for readability

print(f"Dataset information saved to {output_file}")

# %%
