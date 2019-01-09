from PIL import Image
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# # Load the csv file containing the training data
# f = pd.read_csv("fashion-mnist_train.csv")
# df = pd.DataFrame(f)

# print(df.head())
# print(df.shape)

# X = []
# Y = []

# for i in range(0,len(df)):
#     print(i)
#     l = []
#     for j in range(1,len(df.iloc[i])):
#         l.append(df.iloc[i][j])    # Extract the pixel values
#     X.append(l)
#     Y.append(int(df.iloc[i][0]))   # Extract the label

# X = np.array(X)
# Y = np.array(Y)

# print(X.shape)
# print(X[0].shape)

# # Reshape the image into the format accepted by tensorflow
# X = X.reshape(-1, 28, 28)

# Y_final = []

# for i in range(0,len(Y)):
#     if Y[i] == 0:
#         Y_final.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#     elif Y[i] == 1:
#         Y_final.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
#     elif Y[i] == 2:
#         Y_final.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
#     elif Y[i] == 3:
#         Y_final.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
#     elif Y[i] == 4:
#         Y_final.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
#     elif Y[i] == 5:
#         Y_final.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
#     elif Y[i] == 6:
#         Y_final.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
#     elif Y[i] == 7:
#         Y_final.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
#     elif Y[i] == 8:
#         Y_final.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
#     elif Y[i] == 9:
#         Y_final.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])


# # Save the arrays to a numpy file
# np.save("X_RNN.npy", X)
# np.save("Y_RNN.npy", Y_final)

X = np.load("X_RNN.npy")
Y_final = np.load("Y_RNN.npy")
Y_final = np.asarray(Y_final)

print("Shapes ----- ")
    
print(X.shape)
print(X[0].shape)
print(Y_final.shape)
print(Y_final[0].shape)

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y_final, test_size=0.3, random_state=42)

print(X_train.shape)
print(Y_train.shape)

# Model formation
model = Sequential()

no_of_units = 10
no_of_steps = 28
no_of_inputs = 28
no_of_outputs = 10
batch_size = 1
no_of_epochs = 20

model.add(LSTM(128, input_shape=(no_of_steps, no_of_inputs)))
model.add(Dense(no_of_outputs, activation='softmax'))   # For the output class
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=no_of_epochs, shuffle=False)
test_loss = model.evaluate(X_valid, Y_valid)
print(test_loss)