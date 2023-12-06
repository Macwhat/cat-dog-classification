import matplotlib.pyplot as plt 
import tensorflow as tf 
import pandas as pd 
import numpy as np 
  
import warnings 
warnings.filterwarnings('ignore') 
  
from tensorflow import keras 
from keras import layers 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense 
from tensorflow.keras.layers import Conv2D, MaxPooling2D 
from tensorflow.keras.utils import image_dataset_from_directory 
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img 
from tensorflow.keras.preprocessing import image_dataset_from_directory 
  
import os 
import matplotlib.image as mpimg

#Importing Dataset
from zipfile import ZipFile 
  
data_path = 'dog-vs-cat-classification.zip'
  
with ZipFile(data_path, 'r') as zip: 
    zip.extractall() 
    print('The data set has been extracted.')
    
#Data Visualization
path = 'dog-vs-cat-classification'
classes = os.listdir(path) 
classes

fig = plt.gcf() 
fig.set_size_inches(16, 16) 
  
cat_dir = os.path.join('dog-vs-cat-classification/cats') 
dog_dir = os.path.join('dog-vs-cat-classification/dogs') 
cat_names = os.listdir(cat_dir) 
dog_names = os.listdir(dog_dir) 
  
pic_index = 210
  
cat_images = [os.path.join(cat_dir, fname) 
              for fname in cat_names[pic_index-8:pic_index]] 
dog_images = [os.path.join(dog_dir, fname) 
              for fname in dog_names[pic_index-8:pic_index]] 
  
for i, img_path in enumerate(cat_images + dog_images): 
    sp = plt.subplot(4, 4, i+1) 
    sp.axis('Off') 
  
    img = mpimg.imread(img_path) 
    plt.imshow(img) 
  
plt.show()

base_dir = 'dog-vs-cat-classification'
  
# Create datasets 
train_datagen = image_dataset_from_directory(base_dir, 
                                                  image_size=(200,200), 
                                                  subset='training', 
                                                  seed = 1, 
                                                 validation_split=0.1, 
                                                  batch_size= 32) 
test_datagen = image_dataset_from_directory(base_dir, 
                                                  image_size=(200,200), 
                                                  subset='validation', 
                                                  seed = 1, 
                                                 validation_split=0.1, 
                                                  batch_size= 32)

