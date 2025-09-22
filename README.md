# Proyecto Red Neuronal MNIST con DnnLib
Este proyecto es una colección de scripts en Python que exploran y entrenan una red neuronal para el dataset MNIST (y Fashion MNIST) utilizando la librería DnnLib. El objetivo es experimentar con redes neuronales para clasificación de imágenes, desde pruebas básicas hasta entrenamiento y evaluación de modelos.
## Archivos principales

### 1. `1DNNLibTrial.py`
**Propósito:**  
Prueba básica de las funcionalidades de la librería DnnLib.  
- Crea capas densas con diferentes activaciones (ReLU, TanH).
- Muestra cómo funcionan los métodos `forward` y `forward_linear`.
- Prueba funciones de activación directas como `sigmoid`.
---
### 2. `2DatasetTrial.py`
**Propósito:**
Explora y visualiza el dataset MNIST.
- Carga los datos de entrenamiento y prueba.
- Muestra la forma de los datos.
- Imprime algunas imágenes con sus etiquetas usando matplotlib.
---
### 3. `3RN_Accuracy.py`
**Propósito:**  
Evalúa la precisión de un modelo previamente entrenado.  
- Carga pesos y sesgos desde un archivo JSON.
- Realiza un forward pass sobre el set de entrenamiento.
- Calcula y muestra la precisión del modelo.
---
### 4. `4TrainModel.py`
**Propósito:**  
Entrena una red neuronal sobre MNIST.
- Permite cargar y guardar pesos.
- Entrena el modelo usando mini-batch y el optimizador Adam.
- Muestra la pérdida y precisión
---
### 5. `5FashionModel.py`
**Propósito:**  
Entrena una red neuronal sobre el dataset Fashion MNIST.  
- Similar a `4TrainModel.py` pero usando Fashion MNIST.
- Incluye una capa de Dropout y regularización L2.
- Permite cargar y guardar pesos.
- Muestra la pérdida y precisión
---

## Notas
- Se usan archivos `.npz` para los datos de entrada y `.json` para guardar/cargar pesos.
- Los parámetros (tamaño de batch, épocas, learning rate) se cambian en los scripts.
- Para los archivos con arg parser se encuentran disponibles estas opciones:

  Se puede incluir un archivo, si no se nombra uno el proceso se ejecuta con pesos aleatorios.
  options:  
    -h, --help       show this help message and exit  
    -f, --file FILE  Path to load json file with weights  
    -s, --save SAVE  Save the weights to a json file  

*Documentación realizada con Copilot*
