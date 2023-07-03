"""Este programa reliza la division de una imagen y una constante mayor a 1, cuidando de que no haya saturación. 
Luego la compara con la multiplicacion obtenida por openCV."""

#Importa librerias
import cv2
import numpy as np

#Carga y almacena la imagen
img = cv2.imread("imagen1.jpg")

#Constantes de division
constante = 2.5
constante_ocv = np.ones_like(img) * constante

#Divide por la constante y redondea al entero mas proximo
img_dividida1 = np.round(np.clip(img.astype(np.float32) / constante, 0, 255)).astype(np.uint8)

#Divide usando openCV
img_dividida2 = cv2.divide(img, constante_ocv, dtype=cv2.CV_8U)

#Muetra los resultados
cv2.imshow('Imagen Original', img)
cv2.imshow('Imagen Dividida (con NumPy)', img_dividida1)
cv2.imshow('Imagen Dividida (con OpenCV)', img_dividida2)

#Guarda las imagenes
#cv2.imwrite("dividida1.jpg", img_dividida1)
#cv2.imwrite("divididaocv1.jpg", img_dividida2)

#Espera un input y destruye las ventanas
cv2.waitKey(0)
cv2.destroyAllWindows()
