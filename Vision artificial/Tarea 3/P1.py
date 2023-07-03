#Importa librerias
import cv2
import numpy as np
import os
from Funciones.clear import clear

#Limpia la consola
clear()
#######Comparacion Metodo Otsu#######
 
#Carga la imagen en escala de grises
img1 = cv2.imread(os.path.dirname(__file__) + "./imagen1.jpg", 0)
img2 = cv2.imread(os.path.dirname(__file__) + "./imagen2.jpg", 0)
img3 = cv2.imread(os.path.dirname(__file__) + "./Codigo de ayuda/papagayo.jpg", 0)
img4 = cv2.imread(os.path.dirname(__file__) + "./Codigo de ayuda/objetos.jpg", 0)
img5 = cv2.imread(os.path.dirname(__file__) + "./Codigo de ayuda/Star.jpg", 0)

#Aplica umbralización global utilizando el método de Otsu
_, thresholded_image1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, thresholded_image2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, thresholded_image3 = cv2.threshold(img3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, thresholded_image4 = cv2.threshold(img4, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, thresholded_image5 = cv2.threshold(img5, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#Muestra la imagen original y la imagen umbralizada
cv2.imshow("Original 1 vs. Thresholded", thresholded_image1)
cv2.imshow("Original 2 vs. Thresholded", thresholded_image2)
cv2.imshow("Original 3 vs. Thresholded", thresholded_image3)
cv2.imshow("Original 4 vs. Thresholded", thresholded_image4)
cv2.imshow("Original 5 vs. Thresholded", thresholded_image5)
cv2.imwrite("otsu1.jpg", thresholded_image1)
cv2.imwrite("otsu2.jpg", thresholded_image2)
cv2.imwrite("otsu3.jpg", thresholded_image3)
cv2.imwrite("otsu4.jpg", thresholded_image4)
cv2.imwrite("otsu5.jpg", thresholded_image5)

cv2.waitKey(0)
cv2.destroyAllWindows()
