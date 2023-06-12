
import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron
from caracteristicas_modelo import caracteristicas_modelo
from grafo_linea_desicion import grafo_linea_desicion
from clear import clear

clear()

#Datos XOR
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Matriz de entradas con las combinaciones de valores posibles para dos variables binarias
labels = np.array([0, 1, 1, 0])  # Etiquetas correspondientes a la operación lógica AND

perceptron = Perceptron(2)

iteraciones = 5  # Número de iteraciones para el entrenamiento
perceptron.train(inputs, labels, iteraciones)  # Se entrena el perceptrón AND con los datos de entrada y las etiquetas correspondientes

caracteristicas_modelo(perceptron, inputs, labels, "XOR")  # Se calculan y muestran las características del modelo de perceptrón AND

# Graficar la línea de decisión en un gráfico 2D
w1, b1 = perceptron.weights, perceptron.bias  # Se obtienen los pesos y sesgo del perceptrón AND entrenado
grafo_linea_desicion(w1, b1, inputs, labels)  # Se grafica la línea de decisión para el perceptrón de implicación
