# File: DL_Models/transformer_model.py
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, LayerNormalization, MultiHeadAttention, Dropout, Add
from tensorflow.keras.models import Model
import numpy as np

# Positional Encoding layer to give Transformer a sense of order in the data
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(max_len, d_model)

    def get_config(self):
        return {"d_model": self.d_model, "max_len": self.max_len}

    def positional_encoding(self, position, d_model):
        angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, (
                    2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]


def Transformer_model(input_shape, num_classes, num_heads=8, ff_dim=128, dropout_rate=0.3, num_transformer_blocks=3):
    inputs = Input(shape=input_shape)

    # Positional encoding to inject sequence information
    x = PositionalEncoding(input_shape[-1])(inputs)

    for _ in range(num_transformer_blocks):
        # Multi-head Attention Layer
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[-1])(x, x)
        attention_output = Dropout(dropout_rate)(attention_output)

        # Residual connection and normalization after attention
        x = Add()([x, attention_output])
        x = LayerNormalization(epsilon=1e-6)(x)

        # Feed-forward network
        ffn_output = Dense(ff_dim, activation='relu')(x)
        ffn_output = Dense(input_shape[-1])(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)

        # Residual connection and normalization after feed-forward network
        x = Add()([x, ffn_output])
        x = LayerNormalization(epsilon=1e-6)(x)

    # Flatten and classification head
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# def Transformer_model(input_shape, num_classes, num_heads=8, ff_dim=128, dropout_rate=0.4):
#     inputs = Input(shape=input_shape)
#     x = inputs
#
#     # Transformer block
#     attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[-1])(x, x)
#     attention_output = Dropout(dropout_rate)(attention_output)
#     attention_output = Add()([x, attention_output])
#     attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
#
#     ffn_output = Dense(ff_dim, activation='relu')(attention_output)
#     ffn_output = Dense(input_shape[-1])(ffn_output)
#     ffn_output = Dropout(dropout_rate)(ffn_output)
#     ffn_output = Add()([attention_output, ffn_output])
#     x = LayerNormalization(epsilon=1e-6)(ffn_output)
#
#     x = Flatten()(x)
#     x = Dense(128, activation='relu')(x)
#     x = Dropout(dropout_rate)(x)
#     outputs = Dense(num_classes, activation='softmax')(x)
#
#     model = Model(inputs=inputs, outputs=outputs)
#     return model



# def Transformer_model(input_shape, num_classes, num_heads=8, ff_dim=128, dropout_rate=0.1):
#     inputs = Input(shape=input_shape)
#     x = inputs
#
#     # Transformer block
#     attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[-1])(x, x)
#     attention_output = Dropout(dropout_rate)(attention_output)
#     attention_output = Add()([x, attention_output])
#     attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
#
#     # First feed-forward network (FFN) block
#     ffn_output = Dense(ff_dim, activation='relu')(attention_output)
#     ffn_output = Dropout(dropout_rate)(ffn_output)  # First dropout layer within FFN
#     ffn_output = Dense(input_shape[-1])(ffn_output)
#     ffn_output = Dropout(dropout_rate)(ffn_output)  # Second dropout layer within FFN
#     ffn_output = Add()([attention_output, ffn_output])
#     x = LayerNormalization(epsilon=1e-6)(ffn_output)
#
#     # Classification head
#     x = Flatten()(x)
#     x = Dense(256, activation='relu')(x)  # Increased size of dense layer
#     x = Dropout(dropout_rate)(x)  # Dropout in classification head
#     outputs = Dense(num_classes, activation='softmax')(x)
#
#     model = Model(inputs=inputs, outputs=outputs)
#     return model