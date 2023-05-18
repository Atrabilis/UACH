#Este programa implementa el operador de Robinson en Python

#Importe de librerias
import cv2 
import numpy as np
import os
from Funciones.clear import clear
from Funciones.operador_robinson import operador_robinson

#limpia la consola
clear()

#Lee y almacena la imagen
img = cv2.imread(os.path.dirname(__file__) + '\objetos.jpg')

#Aplica el operador prewitt
robinson= operador_robinson(img)

#Muestra y guarda las imagenes, espera un input y destruye las ventanas
cv2.imshow("Original", img)
cv2.imshow("Operador Robinson", robinson)
cv2.imwrite("robinson.jpg", robinson)
cv2.waitKey(0)
cv2.destroyAllWindows()