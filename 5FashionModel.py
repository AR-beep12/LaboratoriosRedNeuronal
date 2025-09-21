import numpy as np
import DnnLib
import json
import argparse

parser = argparse.ArgumentParser(description="Se puede incluir un archivo, si no se nombra uno el proceso se ejecuta con pesos aleatorios.")
parser.add_argument(
    "-f", "--file",
    type=str,
    help="Path to load json file with weights"
)
parser.add_argument(
    "-s", "--save",
    type=str,
    help="Save the weights to a json file"
)

args = parser.parse_args()

#Cargar Entradas
data = np.load("Datasets/fashion_mnist_train.npz")
images = data ['images']
labels = data ['labels']
entradas = images.reshape(images.shape[0], -1) / 255.0

y = np.zeros((labels.shape[0], 10), dtype=np.float32)
y[np.arange(labels.shape[0]), labels] = 1.0

#Inicializar Capas y Optimizador
capas = [
        DnnLib.DenseLayer(784, 128, DnnLib.ActivationType.RELU),
        DnnLib.DenseLayer(128, 10, DnnLib.ActivationType.SOFTMAX)
    ]

optimizer = DnnLib.Adam(learning_rate=0.001)

def AdjustLayers(nombre):
    try:
        with open (nombre + ".json","r") as f:
            datos = json.load(f)
            
        capas[0].weights = np.array(datos['layers'][0]["W"]).T
        capas[1].weights = np.array(datos['layers'][1]["W"]).T
        capas[0].bias = np.array(datos['layers'][0]["b"]).T
        capas[1].bias = np.array(datos['layers'][1]["b"]).T
        print("Pesos cargados correctamente.")
    except FileNotFoundError:
        print("Archivo no encontrado. Pesos Random")



#Entrenar
def train_minibatch(layers, optimizer, Entradas, y, targets, batch_size=128, epochs=5):
    n_samples = Entradas.shape[0]

    for epoch in range(epochs):
        # Shuffle
        indices = np.random.permutation(n_samples)
        Entradas_shuffled = Entradas[indices]
        y_shuffled = y[indices]
        
        epoch_loss = 0.0
        n_batches = 0
        # Process mini-batches
        for i in range(0, n_samples, batch_size):
            Entradas_batch = Entradas_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            #Forward
            f1 = layers[0].forward(Entradas_batch)
            output = layers[1].forward(f1)
            output_lin = layers[1].forward_linear(f1)
            #Loss
            loss = DnnLib.cross_entropy(output, y_batch)
            #Backpropagation
            grad = DnnLib.softmax_crossentropy_gradient(output_lin, y_batch)
            grad = layers[1].backward(grad)
            grad = layers[0].backward(grad)
            #Update
            for layer in layers:
                optimizer.update(layer)
                
            epoch_loss += loss
            n_batches += 1
        avg_loss = epoch_loss / n_batches
        if epoch % 1 == 0:
            accuracy = test(layers)
            print(f"Epoch {epoch}, Avg Loss: {avg_loss:.6f}, Accuracy: {accuracy}")

def test(layers):
    data = np.load("Datasets/fashion_mnist_test.npz")
    imagesT = data ['images']
    labelsT = data ['labels']
    entradas = imagesT.reshape(imagesT.shape[0], -1) / 255.0
    
    s1 = layers[0].forward(entradas)
    s2 = layers[1].forward(s1)

    predictions = np.argmax(s2, axis=1)
    accuracy = np.mean(predictions == labelsT)
    return accuracy


def guardarEnArchivos(nombre):
    try:
        info = {
            "input_shape": [28,28],
            "preprocess": {"scale": 255.0},
            "layers":[
                {"type":"dense","units": 128,"activation": "relu","W":capas[0].weights.T.tolist(),"b":capas[0].bias.T.tolist()},
                {"type":"dense","units": 10,"activation":"softmax","W":capas[1].weights.T.tolist(),"b":capas[1].bias.T.tolist()}
            ]
        }
        with open(nombre + ".json", "w") as f:
            json.dump(info, f, indent=4)       
        print("Pesos guardados")
    except Exception as e:
        print("Error al guardar los pesos:", e)

if args.file:
    AdjustLayers(args.file)

train_minibatch(capas, optimizer, entradas, y, labels, 128, 10)

if args.save:
    guardarEnArchivos(args.save)