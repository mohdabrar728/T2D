import os
import pandas as pd

# Define the directory containing the files
directory = '/project/workspace/diabetes_project/datasets/Diabetes-Data'  # Replace with the path to your data files

# Define the code mapping
code_mapping = {
    33: "Regular insulin dose",
    34: "NPH insulin dose",
    35: "UltraLente insulin dose",
    48: "Unspecified blood glucose measurement",
    57: "Unspecified blood glucose measurement",
    58: "Pre-breakfast blood glucose measurement",
    59: "Post-breakfast blood glucose measurement",
    60: "Pre-lunch blood glucose measurement",
    61: "Post-lunch blood glucose measurement",
    62: "Pre-supper blood glucose measurement",
    63: "Post-supper blood glucose measurement",
    64: "Pre-snack blood glucose measurement",
    65: "Hypoglycemic symptoms",
    66: "Typical meal ingestion",
    67: "More-than-usual meal ingestion",
    68: "Less-than-usual meal ingestion",
    69: "Typical exercise activity",
    70: "More-than-usual exercise activity",
    71: "Less-than-usual exercise activity",
    72: "Unspecified special event",
}

# Initialize a list to store data from all files
all_data = []

# Loop through each file matching the naming pattern
for file_name in sorted(os.listdir(directory)):
    if file_name.startswith("data-") and not os.path.isdir(file_name):  # Check file naming pattern
        file_path = os.path.join(directory, file_name)
        # Read the file
        with open(file_path, 'r') as file:
            for line in file:
                # Split the line into fields
                fields = line.strip().split('\t')
                # Map the code field to its description
                fields[2] = code_mapping.get(int(fields[2]), f"Unknown code: {fields[2]}")
                all_data.append(fields)

# Convert the collected data into a Pandas DataFrame
df = pd.DataFrame(all_data, columns=['Date', 'Time', 'Code', 'Value'])

# Save the combined data to an Excel file
output_path = '/project/workspace/diabetes_project/datasets/merged_data_with_mapping.xlsx'  # Replace with the desired output path
df.to_excel(output_path, index=False)

print(f"Data merged and mapped successfully into {output_path}")
