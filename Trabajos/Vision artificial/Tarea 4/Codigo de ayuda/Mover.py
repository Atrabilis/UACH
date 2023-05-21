import numpy as np
import cv2 as cv
import os
from matlab_functions import bwareaopen, bwlabel

A = cv.imread(os.path.dirname(__file__) + '\objetos.jpg')
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

#Matriz con coordenadas (fil,col) de bordes
Bordes= np.zeros((1, 2), int)
temp = np.zeros((1, 2), int)

indice = 0

# Barrido de la Matriz

for fil in range (1, s[0] - 1):
    for col in range (1, s[1] - 1):
        # Busca Bordes Verticalmente
        if Aetiq[fil,col] == objeto and (Aetiq[fil - 1, col] == 0 or Aetiq[fil + 1, col] == 0):
           Bordes[indice, :] = [fil, col]
           Bordes = np.append(Bordes, temp, axis = 0)
           indice += 1
        
        # Busca Bordes Horizontalmente
        if Aetiq[fil,col] == objeto and (Aetiq[fil,col - 1] == 0 or Aetiq[fil,col + 1] == 0):
           Bordes[indice, :] = [fil, col]
           Bordes = np.append(Bordes, temp, axis = 0)
           indice += 1
            
Bordes = np.delete(Bordes, indice, 0)

while True:

    Abordes = np.zeros((s[0], s[1]), dtype = 'uint8')

    # Creación de Matriz de Bordes
    for i in range(len(Bordes[:,0])):
        Abordes[Bordes[i,0], Bordes[i,1]] = 1
   
   # Actualiza imagen
    cv.imshow("Objeto movil", 255*Abordes)
    cv.setWindowProperty("Objeto movil", cv.WND_PROP_TOPMOST, 1)
    cv.waitKey(1)

    direccion = int(input('En qué dirección desea mover el objeto: \n 1 = Arriba \n 2 = Abajo \n 3 = Izquierda \n 4 = Derecha \n 5 = Detener \n'))
    os.system('cls')

    # Ejecuta el movimiento
    if direccion == 1: Bordes[:,0] = Bordes[:,0] - X
    if direccion == 2: Bordes[:,0] = Bordes[:,0] + X
    if direccion == 3: Bordes[:,1] = Bordes[:,1] - X
    if direccion == 4: Bordes[:,1] = Bordes[:,1] + X
    if direccion == 5: break
    
   
    #Impide que el objeto se 'salga' de la matriz de imagen   
    for k in range(len(Bordes[:,0])):
        if Bordes[k,1] < 0: Bordes[:,1] = Bordes[:,1] + X
        if Bordes[k,1] >= s[1]: Bordes[:,1] = Bordes[:,1] - X
        if Bordes[k,0] < 0: Bordes[:,0] = Bordes[:,0] + X
        if Bordes[k,0] >= s[0]: Bordes[:,0] = Bordes[:,0] - X