#Este codigo codifica cada uno de los objetos de la imagen “objetos.jpg” utilizando
#código cadenay luego crea una nueva imagen en donde los objetos son generados
#a partir del código cadena obtenido anteriormente.
import cv2
import os
import numpy as np
from Funciones.clear import clear
from Funciones.cadena import *

#Limpia la consola
clear()

#Carga la imagen
img = cv2.imread("./Codigo de ayuda/objetos.jpg")

#Crea una nueva imagen con fondo blanco del mismo tamaño que la imagen original
imagen = np.zeros_like(img) + 255

#Convierte la imagen a escala de grises
imagen_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Aplica umbral para binarizar la imagen
umbral, imagen_umbral = cv2.threshold(imagen_gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
imagen_umbral = cv2.bitwise_not(imagen_umbral)

#Encuentra los contornos en la imagen binarizada
contornos = encontrar_contornos(imagen_umbral)

#Convierte los contornos a un formato adecuado
contornos_np = [np.array(contorno) for contorno in contornos]

#Dibuja los contornos
cv2.drawContours(imagen, contornos_np, -1, (0, 0, 255), 2)

#Muestra la imagen con los contornos
cv2.imshow("codificada", imagen)
cv2.imshow("original", img)
cv2.imwrite("cadena.jpg", imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()



