import numpy as np
import cv2 as cv
import os
from Codigo_de_ayuda.matlab_functions import bwareaopen, bwlabel

A = cv.imread(os.path.dirname(__file__) + '.\Codigo_de_ayuda\objetos.jpg')
Agris = cv.cvtColor(A, cv.COLOR_BGR2GRAY)

# Especifica valor de umbral
umbral = 240

# Magnitud del movimiento
X = 10

# Selección de objeto
objeto = 1

#Transforma a Imagen Binaria utilizando el umbral 
ret1, Abin = cv.threshold(Agris,umbral,1,0)

#Filtra Imagen
Abin = bwareaopen(Abin,50)
Abin = 1 - Abin
#Filtra Negativo de la Imagen
Abin = bwareaopen(Abin,50)
Abin = 1-Abin

#Calcula dimensión de matriz de imagen
s = Abin.shape

# Etiqueta objetos
Aetiq = bwlabel(Abin)

# Matriz con coordenadas (fil,col) de bordes
Bordes = []

# Barrido de la Matriz
for fil in range(1, s[0] - 1):
    for col in range(1, s[1] - 1):
        # Busca Bordes
        if Aetiq[fil, col] == objeto:
            if (Aetiq[fil - 1, col] == 0 or Aetiq[fil + 1, col] == 0 or
                    Aetiq[fil, col - 1] == 0 or Aetiq[fil, col + 1] == 0):
                Bordes.append((fil, col))

# Crea una matriz de bordes
Abordes = np.zeros((s[0], s[1]), dtype='uint8')
for punto in Bordes:
    Abordes[punto] = 1

# Muestra la imagen con los bordes del objeto
cv.imshow("Objeto movil", 255 * Abordes)
cv.setWindowProperty("Objeto movil", cv.WND_PROP_TOPMOST, 1)
cv.waitKey(1)

while True:
    direccion = int(input('En qué dirección desea mover el objeto: \n 1 = Arriba \n 2 = Abajo \n 3 = Izquierda \n 4 = Derecha \n 5 = Detener \n'))
    os.system('cls')

    # Verifica si el objeto se encuentra en los límites de la ventana antes de realizar el movimiento
    if direccion == 1 and min([p[0] for p in Bordes]) >= X:
        for i in range(len(Bordes)):
            Bordes[i] = (Bordes[i][0] - X, Bordes[i][1])
    elif direccion == 2 and max([p[0] for p in Bordes]) + X < s[0]:
        for i in range(len(Bordes)):
            Bordes[i] = (Bordes[i][0] + X, Bordes[i][1])
    elif direccion == 3 and min([p[1] for p in Bordes]) >= X:
        for i in range(len(Bordes)):
            Bordes[i] = (Bordes[i][0], Bordes[i][1] - X)
    elif direccion == 4 and max([p[1] for p in Bordes]) + X < s[1]:
        for i in range(len(Bordes)):
            Bordes[i] = (Bordes[i][0], Bordes[i][1] + X)
    elif direccion == 5:
        break

    # Crea una nueva matriz de bordes actualizada
    Abordes = np.zeros((s[0], s[1]), dtype='uint8')
    for punto in Bordes:
        Abordes[punto] = 1

    # Muestra la imagen con el contorno del objeto actualizado
    cv.imshow("Objeto movil", 255 * Abordes)
    cv.setWindowProperty("Objeto movil", cv.WND_PROP_TOPMOST, 1)
    cv.waitKey(1)

cv.destroyAllWindows()
