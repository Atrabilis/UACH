from perceptron import Perceptron
from caracteristicas_modelo import caracteristicas_modelo
from grafo_linea_desicion import grafo_linea_desicion
import numpy as np
import matplotlib.pyplot as plt

#Datos
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
labelsAND = np.array([0,0,0,1])
labelsIMPLICA = np.array([1,1,0,1])

#Instancia de perceptrones
perceptronAND = Perceptron(2)
perceptronIMPLICA = Perceptron(2)

#Aprendizaje perceptrones
iteraciones = 5
perceptronAND.train(inputs, labelsAND, iteraciones)
perceptronIMPLICA.train(inputs, labelsIMPLICA, iteraciones)

#Caracteristicas Perceptrones
caracteristicas_modelo(perceptronAND, inputs, labelsAND, "AND")
caracteristicas_modelo(perceptronIMPLICA, inputs, labelsIMPLICA, "Implica")

# Gráfico 2D de recta de decisión
w1, b1 = perceptronAND.weights, perceptronAND.bias
w2, b2 = perceptronIMPLICA.weights, perceptronIMPLICA.bias
grafo_linea_desicion(w1, b1, inputs, labelsAND)
grafo_linea_desicion(w2, b2, inputs, labelsIMPLICA)


