import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import cifar10


# Load Data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Normalize the value between 0 and 1. Also we dont reshape or flatten it in the begining WHEN using CNN
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
print(x_train.shape)
print(y_train.shape)

# Lets create a Functional API
def func_model():
    inputs = keras.Input(shape=(32,32,3))
    x = layers.Conv2D(32, 3, padding='same', kernel_regularizer=regularizers.l2(1e-2))(inputs)   # No activation function here because we are using batch normalization
    x = layers.BatchNormalization()(x) 
    x = keras.activations.relu(x)      # Activation function is called after Batch Normalization
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding='same', kernel_regularizer=regularizers.l2(1e-2))(x)
    x = layers.BatchNormalization()(x) 
    x = keras.activations.relu(x) 
    x = layers.Conv2D(128, 3, padding='same', kernel_regularizer=regularizers.l2(1e-2))(x)
    x = layers.BatchNormalization()(x) 
    x = keras.activations.relu(x)
    x = layers.Flatten()(x)     # You flatten before you pass it to the FULLY CONNECTED LAYER(FC layer)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-2))(x)  # FC Layer
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

model = func_model() 
print(model.summary())
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    metrics=['accuracy'],
)

model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)
model.evaluate(x_test, y_test, batch_size=64, verbose=2)