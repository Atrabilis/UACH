#Este codigo utiliza el código Run Length y le permite al usuario mover el objeto
#en 4 direcciones utilizando la siguiente tabla de equivalencias:
#Izquierda → 1
#Abajo → 2
#Arriba → 3 
#Derecha → 4

import numpy as np
import cv2 as cv
import os
from Codigo_de_ayuda.matlab_functions import bwareaopen, bwlabel

def codificar_rle(contorno):
    #Inicializa la lista para almacenar el contorno codificado en RLE
    contorno_codificado = []
    #Inicializa el pixel actual y el conteo
    pixel_actual = contorno[0]
    conteo = 1

    for i in range(1, len(contorno)):
        #Comprueba si el pixel actual es igual al siguiente
        if contorno[i] == pixel_actual:
            #Incrementa el conteo
            conteo += 1
        else:
            #Agrega el par (pixel, conteo) al contorno codificado
            contorno_codificado.append((pixel_actual, conteo))
            #Actualiza el pixel actual y reinicia el conteo
            pixel_actual = contorno[i]
            conteo = 1

    #Agrega el último par (pixel, conteo) al contorno codificado
    contorno_codificado.append((pixel_actual, conteo))
    return contorno_codificado

def decodificar_rle(contorno_codificado):
    #Inicializa la lista para almacenar el contorno decodificado
    contorno_decodificado = []

    for pixel, conteo in contorno_codificado:
        #Agrega el pixel repetido según el conteo al contorno decodificado
        contorno_decodificado.extend([pixel] * conteo)

    return contorno_decodificado

A = cv.imread('./Codigo_de_ayuda/objetos.jpg')
Agris = cv.cvtColor(A, cv.COLOR_BGR2GRAY)

#Especifica el valor de umbral
umbral = 240

#Magnitud del movimiento
X = 10

#Selección del objeto
objeto = 2

#Transforma a imagen binaria utilizando el umbral
ret1, Abin = cv.threshold(Agris, umbral, 1, 0)

#Filtra la imagen
Abin = bwareaopen(Abin, 50)
Abin = 1 - Abin
#Filtra el negativo de la imagen
Abin = bwareaopen(Abin, 50)
Abin = 1 - Abin

#Calcula la dimensión de la matriz de la imagen
s = Abin.shape

#Etiqueta los objetos
Aetiq = bwlabel(Abin)

#Matriz con coordenadas (fil, col) de los bordes
Bordes = []

#Barrido de la matriz
for fil in range(1, s[0] - 1):
    for col in range(1, s[1] - 1):
        #Busca los bordes
        if Aetiq[fil, col] == objeto:
            if (Aetiq[fil - 1, col] == 0 or Aetiq[fil + 1, col] == 0 or
                    Aetiq[fil, col - 1] == 0 or Aetiq[fil, col + 1] == 0):
                Bordes.append((fil, col))

#Codifica el contorno utilizando RLE
contorno = np.concatenate(Bordes).ravel().tolist()
contorno_codificado = codificar_rle(contorno)

#Decodifica el contorno
contorno_decodificado = decodificar_rle(contorno_codificado)

#Convierte las coordenadas decodificadas en una lista de tuplas
Bordes = list(zip(contorno_decodificado[::2], contorno_decodificado[1::2]))

#Muestra la imagen con los bordes del objeto
Abordes = np.zeros((s[0], s[1]), dtype='uint8')
for punto in Bordes:
    Abordes[punto] = 1

cv.imshow("Objeto movil", 255 * Abordes)
cv.setWindowProperty("Objeto movil", cv.WND_PROP_TOPMOST, 1)
cv.waitKey(1)

while True:
    direccion = int(input('En qué dirección desea mover el objeto: \n 1 = Arriba \n 2 = Abajo \n 3 = Izquierda \n 4 = Derecha \n 5 = Detener \n'))
    os.system('cls')

    #Verifica si el objeto se encuentra en los límites de la ventana antes de realizar el movimiento
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

    #Crea una nueva matriz de bordes actualizada
    Abordes = np.zeros((s[0], s[1]), dtype='uint8')
    for punto in Bordes:
        Abordes[punto] = 1

    #Muestra la imagen con el contorno del objeto actualizado
    cv.imshow("Objeto movil", 255 * Abordes)
    cv.setWindowProperty("Objeto movil", cv.WND_PROP_TOPMOST, 1)
    cv.waitKey(1)

cv.destroyAllWindows()
