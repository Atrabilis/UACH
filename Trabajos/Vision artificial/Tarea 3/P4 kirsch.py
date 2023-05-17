#Este programa implementa el operador de Kirsch en Python

#Importe de librerias
import cv2 
import numpy as np
from Funciones.clear import clear
from Funciones.operador_kirsch import operador_kirsch


#limpia la consola
clear()

#Lee y almacena la imagen
img = cv2.imread("objetos.jpg")

#Aplica el operador prewitt
kirsch= operador_kirsch(img)

#Muestra y guarda las imagenes, espera un input y destruye las ventanas
cv2.imshow("Original", img)
cv2.imshow("Operador Kirsch", kirsch)
cv2.imwrite("kirsch.jpg", kirsch)
cv2.waitKey(0)
cv2.destroyAllWindows()