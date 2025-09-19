import numpy as np
import DnnLib

entradas = np.array([[0.5, -0.2, 0.1]])

#3 entradas, 5 salidas, con activación ReLU
layer1 = DnnLib.DenseLayer(3, 4, DnnLib.ActivationType.RELU)
layer1.weights = np.array([[0.1, 0.2, 0.3],[0.4, 0.5, 0.6],[0.1, 0.02, -0.3],[0.24, 0.41, 0.3]])
layer1.bias = np.array([0.01, -0.02, 0.022, -0.04])

print("3 entradas, 5 salidas, con activación ReLU")

# Con activación
y1 = layer1.forward(entradas)
print("Salida con activación:", y1)

# lineal, sin activación
y_lin = layer1.forward_linear(entradas)
print("Salida lineal:", y_lin)

# Activaciones directamente
print("Sigmoid:", DnnLib.sigmoid(np.array([0.0, 2.0, -1.0])))

#3 entradas, 2 salidas, con activación TanH
layer = DnnLib.DenseLayer(3, 2, DnnLib.ActivationType.TANH)
layer.weights = np.array([[0.1, 0.2, 0.3],[0.4, 0.5, 0.6]])
layer.bias = np.array([0.01, -0.02])

print("\n3 entradas, 2 salidas, con activación TanH")
y = layer.forward(entradas)
print("Salida con activación:", y)
y_lin = layer.forward_linear(entradas)
print("Salida lineal:", y_lin)