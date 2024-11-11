import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from LSTM_Model import create_lstm_model  # Assuming this file contains the LSTM model architecture

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Directory paths for train, validation, and test sets
train_data_folder = "Data/train"
val_data_folder = "Data/validation"

# Sliding window parameters
window_size = 100  # Number of timesteps in each window
stride = 50  # How much to slide the window after each step

# Ensure a folder for saving graphs exists
plots_folder = "plots"
os.makedirs(plots_folder, exist_ok=True)

# Function to extract sliding windows from a sequence
def extract_sliding_windows(sequence, window_size, stride):
    windows = []
    for start in range(0, len(sequence) - window_size + 1, stride):
        window = sequence[start:start + window_size]
        windows.append(window)
    return np.array(windows)

# Function to load data from a folder and apply sliding windows
def load_data_with_sliding_window(folder_path, window_size, stride):
    all_windows = []
    all_labels = []

    # Loop through each CSV file in the folder
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            print(f"Processing file: {file_path}")

            # Load the CSV file into a DataFrame
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

            # Ensure the sequence has the correct number of frames (500)
            if len(df) == 500:
                # Extract features and label
                if 'Label' in df.columns:
                    label = df['Label'].iloc[0]  # Assuming the label is the same for all rows
                    df = df.drop(columns=['Label'])  # Remove the 'Label' column for features
                else:
                    print(f"Skipping {file_path}: 'Label' column not found.")
                    continue

                # Convert DataFrame to a NumPy array
                sequence = df.values

                # Extract sliding windows from the sequence
                windows = extract_sliding_windows(sequence, window_size, stride)
                if len(windows) == 0:
                    print(f"No windows extracted from {file_path}.")
                else:
                    # Append the windows and their corresponding label
                    all_windows.append(windows)
                    all_labels.extend([label] * len(windows))  # Each window gets the same label
            else:
                print(f"Skipping {file_path}: Invalid sequence length {len(df)}. Expected 500.")

    # Ensure we have windows to concatenate
    if len(all_windows) == 0:
        print("No valid windows were found.")
        return None, None

    # Stack all windows and labels into arrays
    all_windows = np.vstack(all_windows)
    all_labels = np.array(all_labels)
    return all_windows, all_labels

# Load training data using the sliding window approach
X_train, y_train = load_data_with_sliding_window(train_data_folder, window_size, stride)
if X_train is None or y_train is None:
    print("No valid training data found. Exiting.")
    exit()
else:
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")

# Load validation data using the sliding window approach
X_val, y_val = load_data_with_sliding_window(val_data_folder, window_size, stride)
if X_val is None or y_val is None:
    print("No valid validation data found. Exiting.")
    exit()
else:
    print(f"Validation data shape: {X_val.shape}, {y_val.shape}")

# Determine the number of timesteps and features from the training data
timesteps = X_train.shape[1]  # The number of time steps per sequence (window size)
num_features = X_train.shape[2]  # The number of features per time step (should be 23)
num_classes = len(set(y_train))  # Number of unique classes (behavior labels)

# Create the LSTM model using the architecture defined in LSTM_Model.py
model = create_lstm_model(timesteps, num_features, num_classes)

# Define early stopping to prevent overfitting during training
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the LSTM model using the training set and validation set
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=2,  # Adjust epochs as necessary
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Save the trained model
model.save("lstm_ship_behavior_multiclass_model_with_sliding_window.h5")
print("Model saved as 'lstm_ship_behavior_multiclass_model_with_sliding_window.h5'")

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
accuracy_plot_path = os.path.join(plots_folder, "accuracy_plot.png")
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
loss_plot_path = os.path.join(plots_folder, "loss_plot.png")
plt.savefig(loss_plot_path)
print(f"Loss plot saved as {loss_plot_path}")
plt.close()
