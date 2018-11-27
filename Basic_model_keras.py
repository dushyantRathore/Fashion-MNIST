# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten
from keras.layers.core import Dense
from sklearn.model_selection import train_test_split
import keras
import pandas as pd
import numpy as np

df = pd.DataFrame(pd.read_csv("fashion-mnist_train.csv"))
num_classes = 10

# X = []
# Y = []
#
# for i in range(0,10000):
#     print(i)
#     l = []
#     for j in range(1,len(df.iloc[i])):
#         l.append(df.iloc[i][j])    # Extract the pixel values
#     X.append(l)
#     Y.append(int(df.iloc[i][0]))   # Extract the label
#
# X = np.array(X)
# Y = np.array(Y)
#
# np.save("pixel_data.npy", X)
# np.save("labels.npy", Y)


# Load from the save npy files

X = np.load("pixel_data.npy")
Y = np.load("labels.npy")

print("Type ----->")
print(type(X))
print(type(Y))

print("Length ----->")
print(len(X))
print(len(Y))

print("Shape ----->")
print(X.shape)
print(Y.shape)

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3)

# Get the shape of the training set
print(X_train.shape)
print(X_train[0].shape)

# Perform reshaping
X_train = X_train.reshape([-1,28,28,1])
X_test = X_test.reshape([-1,28,28,1])

print(X_train.shape)
print(X_train[0].shape)

# Build the model

model = Sequential()    # 32,64,128 and a fully connected layer with number of output classes
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'), )
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3,3), activation='relu'),)
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# Fit the model on the sets
model.fit(X_train, y_train, batch_size=25, epochs=20, verbose=1, validation_data=(X_test, y_test))

# Evaluate the model
score = model.evaluate(X_test, y_test, verbose=1)
print("Score : " + str(score))
print('Test loss:', score[0])
print('Test accuracy:', score[1])