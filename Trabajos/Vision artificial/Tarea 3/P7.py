#Este programa calcula el centro de masa de un objeto y luego lo
#compara con el centro de masa obtenido mediante la funcion
#moments de openCV

#importa librerias
import cv2
import numpy as np
import os
from Funciones.clear import clear

#Limpia la consola
clear()

#Lee la imagen y la convierte a escala de grises
img = cv2.imread(os.path.dirname(__file__) + '\objetos.jpg')
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Aplica umbralización para obtener una imagen binaria
_, umbral = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY)

#Encuentra los contornos en la imagen binaria
contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#Obtiene el primer contorno
contorno = contornos[0]

#Calcula el centro de masa utilizando moments de OpenCV
M = cv2.moments(contorno)
cx_opencv = int(M['m10'] / M['m00'])  # Coordenada x del centro de masa
cy_opencv = int(M['m01'] / M['m00'])  # Coordenada y del centro de masa
centro_opencv = (cx_opencv, cy_opencv)

#Calcula el centro de masa manualmente
cx_manual = int(np.mean(contorno[:, :, 0]))  # Coordenada x promedio del contorno
cy_manual = int(np.mean(contorno[:, :, 1]))  # Coordenada y promedio del contorno
centro_manual = (cx_manual, cy_manual)

#Imprime los resultados
print("Centro de masa obtenido mediante moments de OpenCV:", centro_opencv)
print("Centro de masa calculado manualmente:", centro_manual)