import numpy as np
import os
from backpropagation import Backprop

os.system("cls")

# Definir la arquitectura de la red y las funciones de transferencia
red_arquitectura = [1, 2, 1]  # 9 entradas, 3 neuronas en la primera capa oculta, 2 en la segunda y 1 en la capa de salida
red_tfn = [2, 3]  # Función de transferencia lineal para todas las capas

# Crear una instancia de la clase Backprop con la arquitectura de la red y las funciones de transferencia
bp = Backprop(red_arquitectura, red_tfn)
bp.alfa = 0.001
bp.num_epochs = 10000
bp.target_error = 1E-5


# Datos de entrenamiento
P = np.linspace(-2, 2, 6).reshape(-1, 1).T
T = np.sin(np.pi * P / 4)
print(P.shape)


# Entrenar la red
bp.train(P, T)

# Probar la red
print(bp.sim(P))  # Salida de la red para la entrada de prueba

# Desplegar los pesos y ganancias de la red entrenada
print("Pesos de la red entrenada:")
for i in range(1, len(red_arquitectura)):
    print("Capa", i, ":")
    print(bp.W[i])
    print()

print("Ganancias de la red entrenada:")
for i in range(1, len(red_arquitectura)):
    print("Capa", i, ":")
    print(bp.B[i])
    print()

# Desplegar el número de épocas de entrenamiento
print("Número de épocas de entrenamiento:", bp.num_epochs)

# Desplegar el error medio cuadrático total obtenido luego del entrenamiento
# Esto se muestra en el gráfico de entrenamiento que se genera en el método train()

# Comprobar que la red clasifica correctamente los ejemplos de entrenamiento
for i in range(P.shape[1]):
    print("Entrada:", P[:, i])
    print("Salida esperada:", T[:, i])
    print("Salida de la red:", bp.sim(P[:, [i]]))
    print()

print(bp.target_error <= 1E-5)