import cv2
import numpy as np
from Funciones.clear import clear

clear()

#######Metodo Otsu#######
 
# Cargar la imagen en escala de grises
image1 = cv2.imread('./imagen1.jpg', 0)
image2 = cv2.imread('./imagen2.jpg', 0)
image3 = cv2.imread('./Codigo de ayuda/papagayo.jpg', 0)
image4 = cv2.imread('./Codigo de ayuda/objetos.jpg', 0)
image5 = cv2.imread('./Codigo de ayuda/Star.jpg', 0)

# Aplicar umbralización global utilizando el método de Otsu
_, thresholded_image1 = cv2.threshold(image1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, thresholded_image2 = cv2.threshold(image2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, thresholded_image3 = cv2.threshold(image3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, thresholded_image4 = cv2.threshold(image4, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, thresholded_image5 = cv2.threshold(image5, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

concatenated_image1 = np.hstack((image1, thresholded_image1))
concatenated_image2 = np.hstack((image2, thresholded_image2))
concatenated_image3 = np.hstack((image3, thresholded_image3))
concatenated_image4 = np.hstack((image4, thresholded_image4))
concatenated_image5 = np.hstack((image5, thresholded_image5))

# Mostrar la imagen original y la imagen umbralizada
cv2.imshow('Original 1 vs. Thresholded', concatenated_image1)
cv2.imshow('Original 2 vs. Thresholded', concatenated_image2)
cv2.imshow('Original 3 vs. Thresholded', concatenated_image3)
cv2.imshow('Original 4 vs. Thresholded', concatenated_image4)
cv2.imshow('Original 5 vs. Thresholded', concatenated_image5)
#cv2.imshow('Original 5 ', image5)
#cv2.imshow('Thresholded 5 ', thresholded_image5)

cv2.waitKey(0)
cv2.destroyAllWindows()
