import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist


# Load Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalize the value between 0 and 1. Also we dont reshape or flatten it in the begining WHEN using CNN
x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255.0

# Create custom layer
class Dense(layers.Layer):
    def __init__(self, units):
        super(Dense, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name = 'w',
            shape = (input_shape[-1], self.units),
            initializer ='random_normal',
            trainable = True,
        )

        self.b = self.add_weight(
            name='b', 
            shape=(self.units, ), 
            initializer='zeros',
            trainable = True,
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
    

# Create our custom activation function class
class MyReLU(layers.Layer):
    def __init__(self):
        super(MyReLU, self).__init__()

    def call(self, x):
        return tf.math.maximum(x, 0)

# Create custom Model
class MyModel(keras.Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()
        self.dense1 = Dense(64)
        self.dense2 = Dense(num_classes)
        self.relu = MyReLU()
        # self.dense2 = layers.Dense(num_classes)

    def call(self, input_tensor):
        # x = tf.nn.relu(self.dense1(input_tensor))
        x = self.relu(self.dense1(input_tensor))
        return self.dense2(x)
    

model = MyModel()

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy'],
)


model.fit(x_train, y_train, batch_size=32, epochs=3, verbose=2)

print(model.summary())

model.evaluate(x_test, y_test, batch_size=32, verbose=2)
