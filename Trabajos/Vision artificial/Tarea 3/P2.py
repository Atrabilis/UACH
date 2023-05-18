#Este programa implementa el operador Roberts en Python

#Importe de librerias
import cv2 
import numpy as np
import os
from Funciones.clear import clear
from Funciones.operador_roberts import operador_roberts

#limpia la consola
clear()

#Lee y almacena la imagen
img1 = cv2.imread(os.path.dirname(__file__) + "./imagen1.jpg", 0)
img2 = cv2.imread(os.path.dirname(__file__) + "./imagen2.jpg", 0)
img3 = cv2.imread(os.path.dirname(__file__) + "./Codigo de ayuda/papagayo.jpg", 0)
img4 = cv2.imread(os.path.dirname(__file__) + "./Codigo de ayuda/objetos.jpg", 0)
img5 = cv2.imread(os.path.dirname(__file__) + "./Codigo de ayuda/Star.jpg", 0)

#Convierte img a una imagen binaria usando el metodo Otsu implementado en openCV
_, binaria1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, binaria2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, binaria3 = cv2.threshold(img3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, binaria4 = cv2.threshold(img4, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, binaria5 = cv2.threshold(img5, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

roberts1 = operador_roberts(binaria1)
roberts2 = operador_roberts(binaria2)
roberts3 = operador_roberts(binaria3)
roberts4 = operador_roberts(binaria4)
roberts5 = operador_roberts(binaria5)
#Muestra y guarda las imagenes, espera un input y destruye las ventanas
cv2.imshow("Operador Roberts 1", roberts1)
cv2.imshow("Operador Roberts 2", roberts2)
cv2.imshow("Operador Roberts 3", roberts3)
cv2.imshow("Operador Roberts 4", roberts4)
cv2.imshow("Operador Roberts 5", roberts5)
#cv2.imwrite("roberts1.jpg", roberts1)
#cv2.imwrite("roberts2.jpg", roberts2)
#cv2.imwrite("roberts3.jpg", roberts3)
#cv2.imwrite("roberts4.jpg", roberts4)
#cv2.imwrite("roberts5.jpg", roberts5)
cv2.waitKey(0)
cv2.destroyAllWindows()

