import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# Make sure we don't get any GPU errors
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Load Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalize the value between 0 and 1. Also we dont reshape or flatten it in the begining WHEN using CNN
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

model = keras.Sequential()
model.add(keras.Input(shape=(None, 28)))
model.add(
    layers.SimpleRNN(512, return_sequences=True, activation='relu')
)
model.add(layers.SimpleRNN(512, activation='relu'))
model.add(layers.Dense(10))

print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=['accuracy'],
)

# model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)
# model.evaluate(x_test, y_test, batch_size=64, verbose=2)

# GRU
GRU_model = keras.Sequential()
GRU_model.add(keras.Input(shape=(None, 28)))
GRU_model.add(
    layers.GRU(256, return_sequences=True, activation='tanh') # the default activation functioin for RNN'S is tanh
)
GRU_model.add(layers.GRU(256, activation='tanh'))
GRU_model.add(layers.Dense(10))

print(GRU_model.summary())

GRU_model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=['accuracy'],
)

# GRU_model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)
# GRU_model.evaluate(x_test, y_test, batch_size=64, verbose=2)

# LSTM
LSTM_model = keras.Sequential()
LSTM_model.add(keras.Input(shape=(None, 28)))
LSTM_model.add(
    layers.LSTM(256, return_sequences=True, activation='tanh') # the default activation functioin for RNN'S is tanh
)
LSTM_model.add(layers.LSTM(256, activation='tanh'))
LSTM_model.add(layers.Dense(10))

print(LSTM_model.summary())

LSTM_model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=['accuracy'],
)

# LSTM_model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)
# LSTM_model.evaluate(x_test, y_test, batch_size=64, verbose=2)

# Birectional
bidirectional_model = keras.Sequential()
bidirectional_model.add(keras.Input(shape=(None, 28)))
bidirectional_model.add(
    layers.Bidirectional(layers.LSTM(256, return_sequences=True, activation='tanh'))
)
bidirectional_model.add(
    layers.Bidirectional(layers.LSTM(256, activation='tanh'))
    )
bidirectional_model.add(layers.Dense(10))

print(bidirectional_model.summary())

bidirectional_model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=['accuracy'],
)

bidirectional_model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)
bidirectional_model.evaluate(x_test, y_test, batch_size=64, verbose=2)
