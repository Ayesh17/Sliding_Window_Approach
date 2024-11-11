from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking


def create_lstm_model(timesteps, num_features):
    model = Sequential()
    # Add a masking layer to ignore zero values in the input
    model.add(Masking(mask_value=0.0, input_shape=(timesteps, num_features)))

    # LSTM layers
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))

    # Output layer for binary classification
    model.add(Dense(1, activation='sigmoid'))  # Use 'sigmoid' for binary classification

    # Compile the model with binary crossentropy loss
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
