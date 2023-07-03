"""Este programa reliza la mulñtiplicacion de una imagen y una constante mayor a 1, cuidando de que no haya saturación. 
Luego la compara con la multiplicacion obtenida por openCV."""

#importa librerias
import cv2
import numpy as np

#Lee y almacena la imagen
img = cv2.imread("imagen1.jpg")

#Constantes de multiplicacion
constante = 1.8
constante_ocv = np.ones_like(img) * constante

#Multiplica por una constante y redondea al entero mas proximo
img_multiplicada1 = np.round(np.clip(img.astype(np.float32) * constante, 0, 255)).astype(np.uint8)

#Se asegura que los valores en los canales no presenten saturacion
img_multiplicada1 = np.clip(img_multiplicada1, 0, 255)

#Compara con openCV
img_multiplicada2 = cv2.multiply(img, constante_ocv, dtype=cv2.CV_8U)

#Muestra la imagen original y ambas imágenes multiplicadas
cv2.imshow('Imagen Original', img)
cv2.imshow('Imagen Multiplicada (con NumPy)', img_multiplicada1)
cv2.imshow('Imagen Multiplicada (con OpenCV)', img_multiplicada2)

#Guarda las imagenes
#cv2.imwrite("multiplicada1.jpg", img_multiplicada1)
#cv2.imwrite("multiplicadaocv1.jpg", img_multiplicada2)

#Espera un input y destruye las ventanas
cv2.waitKey(0)
cv2.destroyAllWindows()