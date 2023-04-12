import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from matplotlib import pyplot as plt
from numpy import expand_dims
import numpy as np

model=tf.keras.models.Sequential([
    
    tf.keras.layers.Conv2D(8,(3,3),activation ='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(16,(3,3),activation ='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(32,(3,3),activation ='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(1024,activation='relu'),
    tf.keras.layers.Dense(512,activation='relu'),
    
    tf.keras.layers.Dense(3,activation='softmax')    
 ])

# Model summary
print(model.summary())

# Getting names of layers
layer_names = [layer.name  for layer in model.layers ]
print(layer_names)

# Getting output of the layers
layer_outputs = [layer.output for layer in model.layers]
print(layer_outputs)


feature_map_model = tf.keras.models.Model(inputs=model.inputs, outputs=layer_outputs)

koala_img = 'Koala-min.jpg'
malaria_img = 'malarai1t.jpeg'
# image = load_img(koala_img, target_size=(224,224))

# # convert the image to an array
# image = img_to_array(image)
# # expand dimensions so that it represents a single 'sample'
# image = expand_dims(image, axis=0)
# print(image.shape)
# input /= 255.0
# print(input)
img = load_img(koala_img, target_size=(150, 150))  
input = img_to_array(img)                           
input = input.reshape((1,) + input.shape)                   
input /= 255.0
print(input)

# generate feature maps 
# feed the input into the model
feature_maps = feature_map_model.predict(input)

# decode the feature_maps content.
for layer_name, feature_map in zip(layer_names, feature_maps):
    print(f"The shape of the {layer_name} is =======>> {feature_map.shape}")

'''
# We need to generate feature maps of only convolution layers 
# and not dense layers and hence we will generate feature maps 
# of layers that have “dimension=4″.
image_belt = []
for layer_name, feature_map in zip(layer_names, feature_maps):  
   if len(feature_map.shape) == 4:
      k = feature_map.shape[-1]  
      size=feature_map.shape[1]
      for i in range(k):
        # Standardization and Normalization 
        # of an image to make it palatable to human eyes:-
        feature_image = feature_map[0, :, :, i]
        feature_image-= feature_image.mean()
        feature_image/= feature_image.std ()
        feature_image*=  64
        feature_image+= 128
        feature_image= np.clip(input, 0, 255).astype('uint8')
        image_belt[:, i * size : (i + 1) * size] = feature_image 
        # Finally let us display the image_belts we have generated:-
        scale = 20. / k
        plt.figure( figsize=(scale * k, scale) )
        plt.title ( layer_name )
        plt.grid  ( False )
        plt.imshow( image_belt, aspect='auto')
'''