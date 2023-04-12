import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# optional. use code below only when having trouble using GPU
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Load Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(y_train.shape)

# flatten the features of x_train and normalize the value between 0 and 1
x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.0
print(x_train.shape)
print(y_train.shape)


# Sequential API(Very convenient, not very flexible)
model = keras.Sequential(
    [
        keras.Input(shape=(28*28)),  # this is need before you can call model.summary()
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10),
    ]
)

print(model.summary())

# Second method for creating Sequential Model
model_method2 = keras.Sequential()
# add one layer at a time . ie
model_method2.add(keras.Input(shape=(28*28)))
model_method2.add(layers.Dense(512, activation='relu'))
model_method2.add(layers.Dense(256, activation='relu'))
model_method2.add( layers.Dense(10))

print("second method of creating sequential api summary")
print(model_method2.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=['accuracy'],
)

# model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
# model.evaluate(x_test, y_test, batch_size=32, verbose=2)

# Functional API( A bit more flexible)
inputs = keras.Input(shape=(28*28))
x = layers.Dense(512, activation='relu')(inputs) # we can add the keyword argument name=layer_name
# eg x = layers.Dense(512, activation='relu', name='first_layer)(inputs)
x = layers.Dense(256, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)
functional_model = keras.Model(inputs=inputs, outputs=outputs)

functional_model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), # from_logits is False here since we indicated the activation function softmax at the layer
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=['accuracy'],
)

functional_model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
functional_model.evaluate(x_test, y_test, batch_size=32, verbose=2)