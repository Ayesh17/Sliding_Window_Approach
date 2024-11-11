from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Dropout, SimpleRNN, GRU


def Bidirectional_GRU_model(input_shape, num_classes):
  model = Sequential()
  model.add(Bidirectional(GRU(128, return_sequences=True), input_shape=input_shape))  # Using Bidirectional GRU
  model.add(Dropout(0.2))
  model.add(Bidirectional(GRU(64)))  # Using Bidirectional LSTM
  model.add(Dropout(0.2))
  model.add(Dense(64, activation='relu'))
  model.add(Dense(num_classes, activation='softmax'))
  return model