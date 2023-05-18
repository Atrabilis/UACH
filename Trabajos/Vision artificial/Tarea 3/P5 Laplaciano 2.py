#Este programa implementa el segundo operador Laplaciano visto en clases en Python.

#Importe de librerias
import cv2 
import numpy as np
import os
from Funciones.clear import clear
from Funciones.operador_laplaciano import operador_laplaciano

#limpia la consola
clear()

#Lee y almacena la imagen
img = cv2.imread(os.path.dirname(__file__) + '\objetos.jpg')

#Aplica el operador prewitt
laplaciano= operador_laplaciano(img, 2)

#Muestra y guarda las imagenes, espera un input y destruye las ventanas
cv2.imshow("Original", img)
cv2.imshow("Operador Laplaciano", laplaciano)
cv2.imwrite("laplaciano2.jpg", laplaciano)
cv2.waitKey(0)
cv2.destroyAllWindows()