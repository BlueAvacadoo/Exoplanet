import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('filtered_tic_targets.csv')

# Get the dimensions (returns a tuple: (rows, columns))
dimensions = df.shape

# Print the dimensions
print(f"Dimensions of the CSV (Rows, Columns): {dimensions}")
print(f"Number of rows: {dimensions[0]}")
print(f"Number of columns: {dimensions[1]}")
