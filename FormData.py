# import the necessary packages
import pandas as pd
import numpy as np
import os
import keras
from keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, array_to_img

df = pd.DataFrame(pd.read_csv("fashion-mnist_train.csv"))
num_classes = 10

X = []
Y = []

for i in range(0,10):
    print(i)
    l = []
    for j in range(1,len(df.iloc[i])):
        l.append(df.iloc[i][j])    # Extract the pixel values
    X.append(l)
    Y.append(int(df.iloc[i][0]))   # Extract the label

X = np.array(X)
Y = np.array(Y)

X = np.dstack([X] * 3)

print(X.shape)
print(X[0].shape)

# Reshape the image into the format accepted by tensorflow
X = X.reshape(-1, 28, 28, 3)

print(X.shape)
print(X[0].shape)

# Resize the images 224*224 as required by VGG16
X = np.asarray([img_to_array(array_to_img(im, scale=False).resize((224, 224))) for im in X])
print(X.shape)
print(X[0].shape)

np.save("pixel_data.npy", X)
np.save("labels.npy", Y)