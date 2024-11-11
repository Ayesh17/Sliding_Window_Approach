import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model

# Define the directory for testing CSV files
test_data_folder = "Binary_Data/test"  # Update to the binary-labeled test data folder

# Function to read CSV files and extract sequences and labels
def load_data_from_folder(folder_path):
    sequences = []
    labels = []

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

            # Append the sequence and label to lists
            sequences.append(sequence)
            labels.append(label)

        except pd.errors.EmptyDataError:
            print(f"Skipping {csv_file}: No data in the file (EmptyDataError).")
        except pd.errors.ParserError:
            print(f"Skipping {csv_file}: Parsing error (ParserError).")
        except Exception as e:
            print(f"Skipping {csv_file}: Unexpected error - {e}")

    return np.array(sequences), np.array(labels)

# Load testing data from the 'test' folder
X_test, y_test = load_data_from_folder(test_data_folder)
print(f"Testing data shape: X_test={X_test.shape}, y_test={y_test.shape}")

# Check if any sequences were successfully read
if len(X_test) == 0:
    print("No valid sequences were found. Please check the data directory and file contents.")
    exit()

# Load the trained binary classification model
model = load_model("lstm_ship_behavior_model_binary.h5")  # Update to binary model
print("Model loaded successfully.")

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Predict on the test data
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)  # Convert probabilities to binary classes (threshold = 0.5)

# Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)

# Plot the confusion matrix with custom labels
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['benign', 'hostile'], yticklabels=['benign', 'hostile'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Display classification report with custom labels
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=['benign', 'hostile']))
