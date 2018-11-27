import numpy as np
import os
import keras
from keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, array_to_img


def vgg16():

    X = np.load("pixel_data.npy")
    Y = np.load("labels.npy")

    X = X[0:100]
    Y = Y[0:100]

    print(Y[0])

    Y_final = []

    for i in range(0,len(Y)):
        if Y[i] == 0:
            Y_final.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif Y[i] == 1:
            Y_final.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        elif Y[i] == 2:
            Y_final.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        elif Y[i] == 3:
            Y_final.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        elif Y[i] == 4:
            Y_final.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        elif Y[i] == 5:
            Y_final.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        elif Y[i] == 6:
            Y_final.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        elif Y[i] == 7:
            Y_final.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        elif Y[i] == 8:
            Y_final.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        elif Y[i] == 9:
            Y_final.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    Y_final = np.asarray(Y_final)
    print("Y final shape")
    print(Y_final.shape)
    print(Y_final[0].shape)

    print(X.shape)
    print(X[0].shape)
    print(Y.shape)
    print(Y[0].shape)

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

    model = VGG16(weights='imagenet', include_top=False)

    features = model.predict(X)

    X = features.reshape(len(X), 25088)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y_final, test_size=0.3, random_state=42)

    model = Sequential()

    model.add(Dense(128, input_dim=25088, activation='relu', kernel_initializer='uniform'))
    keras.layers.core.Dropout(0.3, noise_shape=None, seed=None)

    model.add(Dense(256, input_dim=128, activation='sigmoid'))
    keras.layers.core.Dropout(0.4, noise_shape=None, seed=None)

    model.add(Dense(512, input_dim=256, activation='sigmoid'))
    keras.layers.core.Dropout(0.2, noise_shape=None, seed=None)

    model.add(Dense(units=10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    model.fit(X_train, Y_train, epochs=20, batch_size=128, validation_data=(X_valid, Y_valid))


if __name__ == '__main__':

    # append_data()
    vgg16()