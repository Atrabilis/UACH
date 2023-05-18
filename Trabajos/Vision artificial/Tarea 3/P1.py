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

#Concatena las imagenes
concatenated_image1 = np.hstack((img1, thresholded_image1))
concatenated_image2 = np.hstack((img2, thresholded_image2))
concatenated_image3 = np.hstack((img3, thresholded_image3))
concatenated_image4 = np.hstack((img4, thresholded_image4))
concatenated_image5 = np.hstack((img5, thresholded_image5))

#Muestra la imagen original y la imagen umbralizada
cv2.imshow("Original 1 vs. Thresholded", concatenated_image1)
cv2.imshow("Original 2 vs. Thresholded", concatenated_image2)
cv2.imshow("Original 3 vs. Thresholded", concatenated_image3)
cv2.imshow("Original 4 vs. Thresholded", concatenated_image4)
cv2.imshow("Original 5 vs. Thresholded", concatenated_image5)

cv2.waitKey(0)
cv2.destroyAllWindows()
