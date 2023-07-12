import numpy as np
import os
from backpropagation_modified import Backprop

os.system("cls")

# Definir las cuatro matrices binarias originales
m1 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=np.uint8)

m2 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=np.uint8)

m3 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=np.uint8)

m4 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=np.uint8)

# Generar las 20 matrices
matrices = []
for _ in range(5):
    matrices.append(m1.copy())
    matrices.append(m2.copy())
    matrices.append(m3.copy())
    matrices.append(m4.copy())

    num_modificaciones = np.random.randint(2, 6)  # Número aleatorio de modificaciones entre 2 y 5
    filas, columnas = matrices[-1].shape
    for _ in range(num_modificaciones):
        fila_random = np.random.randint(0, filas)
        columna_random = np.random.randint(0, columnas)
        if matrices[-1][fila_random, columna_random] == 0:
            matrices[-1][fila_random, columna_random] = 1
        else:
            matrices[-1][fila_random, columna_random] = 0

print(matrices)
for idx,val in enumerate(matrices):
    matrices[idx] = val.flatten()


# Visualizar las 20 matrices
#for i, matriz in enumerate(matrices):
#    print(f"Matriz {i+1}:")
#    print(matriz)
#    print(matriz.shape)
#    print()

P = np.array(matrices).T
T = np.array([[1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4]])
print(len(T[0]))
#print(matrices)

# Definir la arquitectura de la red y las funciones de transferencia
red_arquitectura = [100, 3,2, 1]  # 100 entradas, 3 neuronas en la primera capa oculta, 2 en la segunda y 1 en la capa de salida
red_tfn = [3, 3, 3]  # Función de transferencia lineal para todas las capas

# Crear una instancia de la clase Backprop con la arquitectura de la red y las funciones de transferencia
bp = Backprop(red_arquitectura, red_tfn)
bp.alfa = 0.00001
bp.num_epochs = 100000
bp.target_error = 1E-8

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

