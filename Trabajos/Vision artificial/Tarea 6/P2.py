# Propósito del código: Implementar y entrenar perceptrones utilizando PyTorch para las operaciones lógicas AND e implicación. 
# Se utilizan los datos de entrada y las etiquetas correspondientes para el entrenamiento, y se calculan las características 
# de los modelos de perceptrón. Además, se grafica la línea de decisión en un gráfico 2D para visualizar los resultados.

from perceptron_torch import Perceptron  # Se importa la clase Perceptron desde el archivo perceptron_torch.py
from grafo_linea_desicion import grafo_linea_desicion  # Se importa la función grafo_linea_desicion desde el archivo grafo_linea_desicion.py
from caracteristicas_modelo import caracteristicas_modelo  # Se importa la función caracteristicas_modelo desde el archivo caracteristicas_modelo.py
from clear import clear
import torch

clear()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Utiliza la GPU (CUDA) si está disponible, de lo contrario, utiliza la CPU

# Datos de entrada
inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32, device=device)  # Tensor de entradas con las combinaciones de valores posibles para dos variables binarias
labelsAND = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=device)  # Tensor de etiquetas correspondientes a la operación lógica AND
labelsIMPLICA = torch.tensor([1, 1, 0, 1], dtype=torch.float32, device=device)  # Tensor de etiquetas correspondientes a la operación lógica de implicación

# Instanciar objetos Perceptron
perceptronAND = Perceptron(2)  # Se crea un objeto Perceptron con 2 entradas
perceptronIMPLICA = Perceptron(2)  # Se crea otro objeto Perceptron con 2 entradas

# Aprendizaje de los perceptrones
iteraciones = 100  # Número de iteraciones para el entrenamiento
perceptronAND.train(inputs, labelsAND, iteraciones)  # Se entrena el perceptrón AND con los datos de entrada y las etiquetas correspondientes
perceptronIMPLICA.train(inputs, labelsIMPLICA, iteraciones)  # Se entrena el perceptrón de implicación con los datos de entrada y las etiquetas correspondientes

# Obtener características de los perceptrones
caracteristicas_modelo(perceptronAND, torch.tensor(inputs.numpy()), torch.tensor(labelsAND.numpy()), "AND")  # Se calculan y muestran las características del modelo de perceptrón AND
caracteristicas_modelo(perceptronIMPLICA, torch.tensor(inputs.numpy()), torch.tensor(labelsIMPLICA.numpy()), "Implica")  # Se calculan y muestran las características del modelo de perceptrón de implicación

# Graficar la línea de decisión en un gráfico 2D
w1, b1 = perceptronAND.weights, perceptronAND.bias  # Se obtienen los pesos y sesgo del perceptrón AND entrenado
w2, b2 = perceptronIMPLICA.weights, perceptronIMPLICA.bias  # Se obtienen los pesos y sesgo del perceptrón de implicación entrenado
grafo_linea_desicion(w1.numpy(), b1.numpy(), inputs.numpy(), labelsAND.numpy())  # Se grafica la línea de decisión para el perceptrón AND
grafo_linea_desicion(w2.numpy(), b2.numpy(), inputs.numpy(), labelsIMPLICA.numpy())  # Se grafica la línea de decisión para el perceptrón de implicación


