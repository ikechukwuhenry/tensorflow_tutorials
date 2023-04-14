import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import pandas as pd


directory = '/Users/mac/Desktop/DATASETS_KAGGLE/data/mnist_images_csv/'
df = pd.read_csv(directory + 'train.csv')

file_paths = df['file_name'].values
labels = df['label'].values

ds_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))

def read_image(image_file, label):
    image = tf.io.read_file(directory + image_file)
    image = tf.image.decode_image(image, channels=1, dtype=tf.float32)
    return image, label

def augment(image, label):
    # Data augmentation here
    return image, label

ds_train = ds_train.map(read_image).map(augment).batch(2)

# for x, y in ds_train:
#     print(x, y)
#     break

for epoch in range(10):
    for X, Y in ds_train:
        # train here
        pass

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

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    metrics=['accuracy'],
)

model.fit(ds_train, epochs=10, verbose=2)