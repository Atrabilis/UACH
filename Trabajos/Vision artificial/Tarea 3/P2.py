#Este programa implementa el operador Roberts en Python

#Importe de librerias
import cv2 
import numpy as np
from Funciones.clear import clear
from Funciones.operador_roberts import operador_roberts

#limpia la consola
clear()

#Lee y almacena la imagen
img = cv2.imread("objetos.jpg",0)

#Convierte img a una imagen binaria usando el metodo Otsu implementado en openCV
_, binaria = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

roberts = operador_roberts(binaria)
print(roberts.shape)

#Muestra y guarda las imagenes, espera un input y destruye las ventanas
cv2.imshow("Original", img)
cv2.imshow('Imagen binaria', binaria)
cv2.imshow('Operador Roberts', roberts)
cv2.imwrite("binariaroberts.jpg", binaria)
cv2.imwrite("roberts.jpg", roberts)
cv2.waitKey(0)
cv2.destroyAllWindows()

