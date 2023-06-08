# Propósito del código: Implementar y entrenar un perceptrón utilizando la biblioteca PyTorch para la operación lógica (A ∨ B) -> (C ∧ D).
# Se utilizan los datos de entrada y las etiquetas correspondientes para el entrenamiento, y se calculan las características del modelo de perceptrón. 
# Además, se imprimen los outputs generados por el perceptrón.

from perceptron_torch import Perceptron  # Se importa la clase Perceptron desde el archivo perceptron_torch.py
from caracteristicas_modelo import caracteristicas_modelo  # Se importa la función caracteristicas_modelo desde el archivo caracteristicas_modelo.py
from clear import clear
import torch

clear()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Se verifica si se puede utilizar la GPU, de lo contrario, se utiliza la CPU

# Datos de entrada y salida
inputs = torch.tensor([
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 1, 1],
    [1, 0, 0, 0],
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [1, 1, 1, 0],
    [1, 1, 1, 1]
], dtype=torch.float32, device=device)  # Los datos de entrada se convierten en tensores de PyTorch
outputs = torch.tensor([1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float32, device=device)  # Las salidas se convierten en un tensor de PyTorch

# Aprendizaje del perceptrón
perceptron = Perceptron(4)  # Se crea un objeto Perceptron con 4 entradas
iteraciones = 1000  # Número de iteraciones para el entrenamiento
perceptron.train(inputs[:12], outputs[:12], iteraciones)  # Se entrena el perceptrón con los primeros 12 datos de entrada y las etiquetas correspondientes

# Obtener características del perceptrón
caracteristicas_modelo(perceptron, inputs[12:], outputs[12:], "(A ∨ B) -> (C ∧ D)")  # Se calculan y muestran las características del modelo de perceptrón

print("\nOutputs del perceptrón")
print(perceptron.forward(inputs[12:]))  # Se imprime los outputs generados por el perceptrón para los últimos 4 datos de entrada

