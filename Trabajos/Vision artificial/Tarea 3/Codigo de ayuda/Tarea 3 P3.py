#**************Operador Sobel*************************************
import math
import cmath
import numpy as np
import cv2 as cv
import os

# Lee imagen
A = cv.imread(os.path.dirname(__file__) + '\objetos.jpg')

# Transforma a escala de grises
Agris = cv.cvtColor(A, cv.COLOR_BGR2GRAY)

# Especifica valor de umbral
umbral = 220
ret1, Abin = cv.threshold(Agris,umbral,1,0)

#Calcula dimensión de matriz de imagen
s = A.shape[:]

#Crea Matriz de Bordes
s1 = 0
s2 = 0
Magnitud = np.zeros((s[0], s[1]))
Direccion = np.zeros((s[0], s[1]))

#Se aplica el operador sobre la imagen
for fil in range (1,s[0] - 1):
    for col in range (1,s[1] - 1):
#Cálculo de S1
        s1 = -1*Abin[fil-1,col-1]-2*Abin[fil-1,col]-Abin[fil-1,col+1] \
        +0*Abin[fil,col-1]+0*Abin[fil,col]+0*Abin[fil,col+1]+Abin[fil+1,col-1] \
        +2*Abin[fil+1,col]+Abin[fil+1,col+1]
    #Cálculo de S2
        s2 = -1*Abin[fil-1,col-1]+0*Abin[fil-1,col]+Abin[fil-1,col+1] \
        -2*Abin[fil,col-1]+0*Abin[fil,col]+2*Abin[fil,col+1]-Abin[fil+1,col-1] \
        +0*Abin[fil+1,col]+Abin[fil+1,col+1]
    #Cálculo de Magnitud
        Magnitud[fil,col] = math.sqrt(s1**2 + s2**2)
    #Cálculo de Dirección, note que para evitar dividir por cero, obtener
    #el arctan de s2/s1 equivale a calcular la dirección del número complejo
    #s1 + is2
        Direccion[fil,col] = cmath.phase(complex(s1, s2))
print(Magnitud)
# Visualiza imágenes
cv.imshow("Imagen Bordes Sobel", Magnitud)
cv.imshow("Imagen Binaria", 255*Abin)
cv.waitKey(0)