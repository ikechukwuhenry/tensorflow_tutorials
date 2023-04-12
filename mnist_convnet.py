import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist


# Load Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalize the value between 0 and 1. Also we dont reshape or flatten it in the begining WHEN using CNN
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
print(x_train.shape)
print(y_train.shape)

# create a sequential model
model = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, padding='same', activation='relu'),   # Padding='valid' is the default and can be ommitted
        layers.MaxPooling2D(pool_size=(2,2)),   # Deffault pool_size is also (2,2) and can be ommited
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(), # Implies pool_size = (2,2)
        layers.Conv2D(128, 3, activation='relu'),  # Implies padding='valid'
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10),   
    ]
)

print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    metrics=['accuracy'],
)

model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)
model.evaluate(x_test, y_test, batch_size=64, verbose=2)