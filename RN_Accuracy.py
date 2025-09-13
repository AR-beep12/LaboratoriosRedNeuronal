import numpy as np
import DnnLib
import json

#Cargar Entradas
data = np.load("Datasets/mnist_train.npz")
test = np.load("Datasets/mnist_test.npz")

imagesD = data ['images']
labelsD = data ['labels']
imagesT = test ['images']
labelsT = test ['labels']

DIma = imagesD.reshape(imagesD.shape[0], -1) / 255.0
TIma = imagesT.reshape(imagesT.shape[0], -1) / 255.0

#Cargar Pesos y Sesgos
layer1 = DnnLib.DenseLayer(784, 128, DnnLib.ActivationType.RELU)
layer2 = DnnLib.DenseLayer(128, 10, DnnLib.ActivationType.SOFTMAX)

with open ("Datasets/mnist_mlp_pretty.json","r") as f:
    datos = json.load(f)
    
capas = []
for elem in datos['layers']:
    capas.append(elem)
    
layer1.weights = np.array(capas[0]["W"]).T
layer2.weights = np.array(capas[1]["W"]).T
layer1.bias = np.array(capas[0]["b"]).T
layer2.bias = np.array(capas[1]["b"]).T

#Forward
salida1 = layer1.forward(DIma)
salida2 = layer2.forward(salida1)

predictions = np.argmax(salida2, axis=1)
accuracy = np.mean(predictions == labelsD)
print("Precisión:", accuracy*100)