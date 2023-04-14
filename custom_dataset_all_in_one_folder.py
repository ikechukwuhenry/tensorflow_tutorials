import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import pathlib

batch_size = 2
img_height = 28
img_width = 28

directory = '/Users/mac/Desktop/DATASETS_KAGGLE/data/mnist_images_csv/'
ds_train = tf.data.Dataset.list_files(str(pathlib.Path(directory + '*.jpg')))
# print(next(iter(ds_train)))

def process_path(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=1)
    label = tf.strings.split(file_path, '/')
    label = tf.strings.substr(label[-1], pos=0, len=1)
    label = tf.strings.to_number(label, out_type=tf.int64)
    return image, label

# for file_path in ds_train:
#     label = tf.strings.split(file_path, '/')
#     label = tf.strings.substr(label[-1], pos=0, len=1)
#     print(label)
#     break

ds_train = ds_train.map(process_path).batch(2)
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