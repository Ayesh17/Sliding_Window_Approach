import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from LSTM_Model import create_lstm_model


# Define the directory containing the CSV files
data_folder = "Data"

# Collect all CSV files in the data folder
csv_files = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith(".csv")]

# Initialize lists to store sequences and labels
all_sequences = []
all_labels = []

# Read each CSV file and extract sequences and labels
for csv_file in csv_files:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Extract features (all columns except 'Label')
    feature_columns = [col for col in df.columns if col != 'Label']
    sequence = df[feature_columns].values

    # Extract label for this sequence (assuming it's the same for all rows in the sequence)
    label = df['Label'].iloc[0]

    # Append the sequence and label to lists
    all_sequences.append(sequence)
    all_labels.append(label)

# Convert the list of sequences into a 3D NumPy array
X = np.array(all_sequences)  # Shape: (number_of_sequences, number_of_timesteps, number_of_features)
y = np.array(all_labels)  # Shape: (number_of_sequences,)

# Check the shapes of the features and labels
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Determine the number of timesteps and features from the input data
timesteps = X.shape[1]  # The number of time steps per sequence
num_features = X.shape[2]  # The number of features per time step
num_classes = len(set(y))  # Number of unique classes (behavior labels)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shapes of training and testing data
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Create the LSTM model using the architecture defined in lstm_model.py
model = create_lstm_model(timesteps, num_features, num_classes)

# Define early stopping to prevent overfitting during training
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the LSTM model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Save the trained model
model.save("lstm_ship_behavior_model.h5")
print("Model saved as 'lstm_ship_behavior_model.h5'")
