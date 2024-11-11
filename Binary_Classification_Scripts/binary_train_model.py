import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from LSTM_Model import create_lstm_model  # Assuming this file contains the LSTM model architecture for binary classification

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Define directories for training and validation CSV files from the binary-labeled dataset
train_data_folder = "Binary_Data/train"
val_data_folder = "Binary_Data/validation"

# Ensure a folder for saving graphs exists
plots_folder = "plots"
os.makedirs(plots_folder, exist_ok=True)

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

# Load training data from the 'train' folder
X_train, y_train = load_data_from_folder(train_data_folder)
print(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")

# Load validation data from the 'validation' folder
X_val, y_val = load_data_from_folder(val_data_folder)
print(f"Validation data shape: X_val={X_val.shape}, y_val={y_val.shape}")

# Check if any sequences were successfully read
if len(X_train) == 0 or len(X_val) == 0:
    print("No valid sequences were found in one or more sets. Please check the data directory and file contents.")
    exit()

# Determine the number of timesteps and features from the training data
timesteps = X_train.shape[1]  # The number of time steps per sequence (should be 500)
num_features = X_train.shape[2]  # The number of features per time step (should be 23)
num_classes = 1  # Binary classification

# Create the LSTM model using the architecture defined in LSTM_Model.py
model = create_lstm_model(timesteps, num_features)

# Define early stopping to prevent overfitting during training
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the LSTM model using the training set and validation set
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),  # Use the validation set for validation
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Save the trained model
model.save("lstm_ship_behavior_model_binary.h5")
print("Model saved as 'lstm_ship_behavior_model_binary.h5'")

# Plot training & validation accuracy values
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.grid(True)

# Save accuracy plot to the graphs folder
accuracy_plot_path = os.path.join(plots_folder, "accuracy_plot_binary.png")
plt.savefig(accuracy_plot_path)
print(f"Accuracy plot saved as {accuracy_plot_path}")
plt.close()

# Plot training & validation loss values
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.grid(True)

# Save loss plot to the graphs folder
loss_plot_path = os.path.join(plots_folder, "loss_plot_binary.png")
plt.savefig(loss_plot_path)
print(f"Loss plot saved as {loss_plot_path}")
plt.close()
