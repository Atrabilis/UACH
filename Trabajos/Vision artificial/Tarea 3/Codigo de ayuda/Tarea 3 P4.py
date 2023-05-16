#**************Operador Máscaras Brújula de Kirsch*************************************

import numpy as np
import cv2 as cv
import os


# Lee imagen
A = cv.imread(os.path.dirname(__file__) + '\objetos.jpg')

# Transforma a escala de grises
Agris = cv.cvtColor(A, cv.COLOR_BGR2GRAY)

# Especifica valor de umbral
umbral = 240
ret1, Abin = cv.threshold(Agris,umbral,1,0)

#Calcula dimensión de matriz de imagen
s = Abin.shape[:]

Magnitud = np.zeros((s[0], s[1]))
Direccion = np.empty((s[0], s[1]), 'U3')

#Crea vector con resultado de convolución con máscaras Kirsch
K = np.zeros((8), int)

#Se aplica el operador sobre la imagen
for fil in range (1,s[0] - 1):
    for col in range (1,s[1] - 1):
    #Cálculo de k0
        K[0] = -3*Abin[fil-1,col-1]-3*Abin[fil-1,col]+5*Abin[fil-1,col+1] \
        -3*Abin[fil,col-1]+0*Abin[fil,col]+5*Abin[fil,col+1]-3*Abin[fil+1,col-1] \
        -3*Abin[fil+1,col]+5*Abin[fil+1,col+1]
    #Cálculo de k1
        K[1] = -3*Abin[fil-1,col-1]+5*Abin[fil-1,col]+5*Abin[fil-1,col+1] \
        -3*Abin[fil,col-1]+0*Abin[fil,col]+5*Abin[fil,col+1]-3*Abin[fil+1,col-1] \
        -3*Abin[fil+1,col]-3*Abin[fil+1,col+1]
    #Cálculo de k2
        K[2] = 5*Abin[fil-1,col-1]+5*Abin[fil-1,col]+5*Abin[fil-1,col+1] \
        -3*Abin[fil,col-1]+0*Abin[fil,col]-3*Abin[fil,col+1]-3*Abin[fil+1,col-1] \
        -3*Abin[fil+1,col]-3*Abin[fil+1,col+1]
    #Cálculo de k3
        K[3] = 5*Abin[fil-1,col-1]+5*Abin[fil-1,col]-3*Abin[fil-1,col+1] \
        +5*Abin[fil,col-1]+0*Abin[fil,col]-3*Abin[fil,col+1]-3*Abin[fil+1,col-1] \
        -3*Abin[fil+1,col]-3*Abin[fil+1,col+1]
    #Cálculo de k4
        K[4] = 5*Abin[fil-1,col-1]-3*Abin[fil-1,col]-3*Abin[fil-1,col+1] \
        +5*Abin[fil,col-1]+0*Abin[fil,col]-3*Abin[fil,col+1]+5*Abin[fil+1,col-1] \
        -3*Abin[fil+1,col]-3*Abin[fil+1,col+1]
    #Cálculo de k5
        K[5] = -3*Abin[fil-1,col-1]-3*Abin[fil-1,col]-3*Abin[fil-1,col+1] \
        +5*Abin[fil,col-1]+0*Abin[fil,col]-3*Abin[fil,col+1]+5*Abin[fil+1,col-1] \
        +5*Abin[fil+1,col]-3*Abin[fil+1,col+1]
    #Cálculo de k6
        K[6] = -3*Abin[fil-1,col-1]-3*Abin[fil-1,col]-3*Abin[fil-1,col+1] \
        -3*Abin[fil,col-1]+0*Abin[fil,col]-3*Abin[fil,col+1]+5*Abin[fil+1,col-1] \
        +5*Abin[fil+1,col]+5*Abin[fil+1,col+1]
    #Cálculo de k7
        K[7] = -3*Abin[fil-1,col-1]-3*Abin[fil-1,col]-3*Abin[fil-1,col+1] \
        -3*Abin[fil,col-1]+0*Abin[fil,col]+5*Abin[fil,col+1]-3*Abin[fil+1,col-1] \
        +5*Abin[fil+1,col]+5*Abin[fil+1,col+1]

    #Cálculo de Magnitud        
        Magnitud[fil,col] = max(K)
        I = np.argmax(K)
        
        #Cálculo de Dirección, si sólo se necesita la Magnitud, se pueden borrar
        #estas líneas para aumentar la velocidad del algoritmo
        #******************************************
        if I == 0: Direccion[fil,col] = "N"
        if I == 1: Direccion[fil,col] = "NO"
        if I == 2: Direccion[fil,col] = "O"
        if I == 3: Direccion[fil,col] = "SO"
        if I == 4: Direccion[fil,col] = "S"
        if I == 5: Direccion[fil,col] = "SE"
        if I == 6: Direccion[fil,col] = "E"
        if I == 7: Direccion[fil,col] = "NE"
        #*******************************************

# Visualiza imágenes
cv.imshow("Imagen Bordes Kirsch", Magnitud)
cv.imshow("Imagen Binaria", 255*Abin)
print(Direccion)
cv.waitKey(0)