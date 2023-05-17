#Este programa implementa el operador Prewitt en Python

#Importe de librerias
import cv2 
import numpy as np
from Funciones.clear import clear
from Funciones.operador_prewitt import operador_prewitt

#limpia la consola
clear()

#Lee y almacena la imagen
img = cv2.imread("objetos.jpg")

#Aplica el operador prewitt
prewitt= operador_prewitt(img)

#Muestra y guarda las imagenes, espera un input y destruye las ventanas
cv2.imshow("Original", img)
cv2.imshow('Operador Prewitt', prewitt)
cv2.imwrite("prewitt.jpg", prewitt)
cv2.waitKey(0)
cv2.destroyAllWindows()