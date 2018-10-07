from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

f = pd.read_csv("fashion-mnist_train.csv")
df = pd.DataFrame(f)

for i in range(0,1):
    pixel_values = []
    for j in range(1,len(df.iloc[0])):  # The first index is the label
        pixel_values.append(df.iloc[i][j])

    print(len(pixel_values))
    pixel_values = np.array(pixel_values)
    pixel_values = pixel_values.reshape(28,28)

    print(pixel_values)
    plt.imshow(pixel_values)
    plt.show()