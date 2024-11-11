import numpy as np
import pandas as pd

# Load the preprocessed dataset from the Data folder
data_path = "Data/preprocessed_ship_behavior_data.npz"
data = np.load(data_path)

# Extract sequences and labels
sequences = data['sequences']
labels = data['labels']

# Select a sample (e.g., the first sequence) to convert back to a CSV
sample_index = 0
sample_sequence = sequences[sample_index]
sample_label = labels[sample_index]

# Convert the sample sequence to a DataFrame
sample_df = pd.DataFrame(sample_sequence)

# Save the sample sequence to CSV for visualization
sample_filename = f"Data/sample_sequence_label_{sample_label}.csv"
sample_df.to_csv(sample_filename, index=False)
print(f"Sample sequence saved as {sample_filename}")
