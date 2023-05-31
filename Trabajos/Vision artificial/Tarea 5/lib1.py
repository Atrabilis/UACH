#Repositorio en github: https://github.com/Atrabilis/UACH/blob/main/Trabajos/Vision%20artificial/Tarea%205/lib1.py

import numpy as np  #Importa la librería numpy para operaciones numéricas

class Neurona:
    def __init__(self, inputs, pesos, bias, funcionActivacion):
        #Inicializa los atributos de la clase Neurona
        self.entradas = inputs
        self.pesos = pesos
        self.bias = bias
        self.funcionActivacion = funcionActivacion
        
    def suma(self):
        #Calcula la suma ponderada de las entradas y pesos utilizando el producto punto de numpy
        return np.dot(self.pesos, self.entradas)
    
    def salida(self):
        #Calcula la salida de la neurona aplicando la función de activación a la suma ponderada
        return self.funcionActivacion(self.suma())
    
def escalon_unitario(suma):
    #Implementa la función de activación escalón unitario
    if suma <= 0:
        return 0
    else:
        return 1

def signo(suma):
    #Implementa la función de activación signo
    if suma <= 0:
        return -1
    else:
        return 1

def sigmoidal(pesos):
    #Implementa la función de activación sigmoide
    return (1 / (1 + np.exp(-pesos)))[0]

#Define las matrices de entrada
X1 = np.reshape([5, 7, 6, 5, 2, -1, 0, -9, -4], (-1, 1))
X2 = np.reshape([2, -4, -8, -9, -6, 1, 7, 0, 5], (-1, 1))
X3 = np.reshape([4, -7, 8, -2, 0, 6, -9, 1, 1], (-1, 1))
X4 = np.reshape([-1, 7, 2, 2, 0, 3, 1, 8, 2], (-1, 1))

#Define los vectores de pesos
pesos1 = [1, -2, 3, -4, 5, -6, 7, -8, 9]
pesos2 = [2, -4, 6, -8, 10, -12, 14, -16, 18]
pesos3 = [-3, 6, -9, 12, -15, 18, -21, 24, -27]
pesos4 = [1, 1, -1]
pesos5 = [1, 2, 2]
pesos6 = [-3, 3]

