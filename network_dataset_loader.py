import pandas as pd

dataset_path = '/Users/karenl/Downloads/network_intrusion_dataset.csv' 
df = pd.read_csv(dataset_path)
print(df.columns)

print("First 5 rows of the dataset:")
print(df.head())


