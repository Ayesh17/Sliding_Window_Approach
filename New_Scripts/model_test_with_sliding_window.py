import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model

# Define the directory for testing CSV files
test_data_folder = "Data/test"

# Behavior label mapping
behavior_mapping = {
    0: "benign",
    1: "block",
    2: "ram",
    3: "cross",
    4: "headon",
    5: "herd",
    6: "overtake"
}

# Sliding window parameters
window_size = 100  # Number of timesteps in each window
stride = 50  # How much to slide the window after each step

# Function to extract sliding windows from a sequence
def extract_sliding_windows(sequence, window_size, stride):
    windows = []
    for start in range(0, len(sequence) - window_size + 1, stride):
        window = sequence[start:start + window_size]
        windows.append(window)
    return np.array(windows)

# Function to read CSV files, extract sliding windows and labels
def load_data_with_sliding_window(folder_path, window_size, stride):
    all_sequences = []  # For full sequences
    all_labels = []     # For labels of sequences
    all_windows = []    # For extracted windows

    # Collect all CSV files in the folder
    csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".csv")]

    for csv_file in csv_files:
        try:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(csv_file)

            # Skip files that are empty or don't have the expected 'Label' column
            if df.empty or 'Label' not in df.columns:
                print(f"Skipping {csv_file}: File is empty or 'Label' column not found.")
                continue

            # Extract feature columns (all columns except 'Label')
            feature_columns = [col for col in df.columns if col != 'Label']
            sequence = df[feature_columns].values  # Convert features to a NumPy array

            # Ensure that the sequence has the right shape: (number_of_timesteps, number_of_features)
            if len(sequence) != 500 or sequence.shape[1] != 23:
                print(f"Skipping {csv_file}: Invalid sequence shape {sequence.shape}. Expected (500, 23).")
                continue

            # Extract the label (assuming it's the same for all rows in the sequence)
            label = df['Label'].iloc[0]

            # Append the full sequence and label to the list (for later majority voting)
            all_sequences.append(sequence)
            all_labels.append(label)

            # Extract sliding windows from the sequence
            windows = extract_sliding_windows(sequence, window_size, stride)
            all_windows.append(windows)

        except pd.errors.EmptyDataError:
            print(f"Skipping {csv_file}: No data in the file (EmptyDataError).")
        except pd.errors.ParserError:
            print(f"Skipping {csv_file}: Parsing error (ParserError).")
        except Exception as e:
            print(f"Skipping {csv_file}: Unexpected error - {e}")

    # Stack all windows into one array
    all_windows = np.vstack(all_windows)
    return np.array(all_sequences), np.array(all_labels), all_windows

# Load test data using the sliding window approach
X_test_full_sequences, y_test, X_test_windows = load_data_with_sliding_window(test_data_folder, window_size, stride)
print(f"Testing windows shape: {X_test_windows.shape}, Testing full sequences shape: {X_test_full_sequences.shape}, Labels shape: {y_test.shape}")

# Check if any windows were successfully read
if len(X_test_windows) == 0:
    print("No valid windows were found. Please check the data directory and file contents.")
    exit()

# Load the trained model
model = load_model("lstm_ship_behavior_multiclass_model_with_sliding_window.h5")
print("Model loaded successfully.")

# Predict on the test data windows
y_pred_windows = model.predict(X_test_windows)
y_pred_classes_windows = np.argmax(y_pred_windows, axis=1)

# Now we need to aggregate predictions for each full sequence using majority voting or averaging
def majority_vote(predictions):
    values, counts = np.unique(predictions, return_counts=True)
    return values[np.argmax(counts)]  # Return the most frequent prediction

# Aggregate predictions for each sequence
y_pred_sequences = []
num_windows_per_sequence = (500 - window_size) // stride + 1  # Calculate number of windows per sequence
for i in range(0, len(y_pred_classes_windows), num_windows_per_sequence):
    sequence_windows = y_pred_classes_windows[i:i + num_windows_per_sequence]
    final_prediction = majority_vote(sequence_windows)
    y_pred_sequences.append(final_prediction)

# Convert list to NumPy array
y_pred_sequences = np.array(y_pred_sequences)

# Evaluate model's performance on the full sequences
conf_matrix = confusion_matrix(y_test, y_pred_sequences)

# Map numeric labels to behavior names
behavior_labels = [behavior_mapping[i] for i in range(len(behavior_mapping))]

# Plot the confusion matrix with behavior labels
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=behavior_labels, yticklabels=behavior_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Display classification report with behavior labels
print("\nClassification Report:")
print(classification_report(y_test, y_pred_sequences, target_names=behavior_labels))
