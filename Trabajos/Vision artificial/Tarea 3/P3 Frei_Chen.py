#Este programa implementa el operador Frei-Chen en Python

#Importe de librerias
import cv2 
import numpy as np
from Funciones.clear import clear
from Funciones.operador_frei_chen import operador_frei_chen

#limpia la consola
clear()

#Lee y almacena la imagen
img = cv2.imread("objetos.jpg")

#Aplica el operador prewitt
frei_chen= operador_frei_chen(img)

#Muestra y guarda las imagenes, espera un input y destruye las ventanas
cv2.imshow("Original", img)
cv2.imshow('Operador Frei Chen', frei_chen)
cv2.imwrite("freichen.jpg", frei_chen)
cv2.waitKey(0)
cv2.destroyAllWindows()