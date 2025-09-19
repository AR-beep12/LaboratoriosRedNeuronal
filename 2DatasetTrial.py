import numpy as np
import matplotlib.pyplot as plt

data = np.load("Datasets/mnist_train.npz")
test = np.load("Datasets/mnist_test.npz")

imagesD = data ['images']
labelsD = data ['labels']

imagesT = test ['images']
labelsT = test ['labels']

DIma = imagesD.reshape(imagesD.shape[0], -1).T / 255.0
TIma = imagesT.reshape(imagesT.shape[0], -1).T / 255.0

def ImprimirIma (images, labels):
    plt.figure(figsize=(3,3))
    for i in range(3):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i], cmap="gray")
        plt.title(f"Label: {labels[i]}")
        plt.axis("off")
    plt.show()


print("Shape Data:", DIma.shape)
ImprimirIma(imagesD,labelsD)
print("Shape Test:", TIma.shape)
ImprimirIma(imagesT,labelsT)

