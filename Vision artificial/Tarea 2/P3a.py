"""Este programa reliza la suma algebraica de dos imagenes, cuidando de que no haya saturación. 
Luego la compara con la suma obtenida por openCV."""

#se importan librerias
import numpy as np
import cv2

#se lee y almacena las imagenes
img1 = cv2.imread("imagen1.jpg")
img2 = cv2.imread("imagen2.jpg")

#Se limita el valor de los pixeles de las imagenes a 200 como máximo, esto para no saturar la imagen en la suma
img1 = np.clip(img1, 0, 200)
img2 = np.clip(img2, 0, 200)

#se normalizan las imagenes a 255
img1_norm = img1 / 255.0
img2_norm = img2 / 255.0

#se realiza la suma directa
img_sum = img1_norm + img2_norm

#se vuelve a multiplicar la matriz por 255, limitando sus valores entre 0 y 255 y ademas dejando el formato
#de la matriz en uint8 para ser leida por openCV
img_sum = np.clip(img_sum * 255, 0, 255).astype(np.uint8)

#se realiza la suma y la suma ponderada con openCV
sumaOCV = cv2.add(img1,img2)
mezcla1 = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)
mezcla2 = cv2.addWeighted(img1, 0.3, img2, 0.7, 0)

#se muestran las imagenes
cv2.imshow("Imagen 1", img1)
cv2.imshow("Imagen 2", img2)
cv2.imshow("Suma", img_sum)
cv2.imshow("Suma con cv2.add()", sumaOCV)
cv2.imshow("Suma Ponderada img1 dominante", mezcla1)
cv2.imshow("Suma Ponderada img2 dominante", mezcla2)

#cv2.imwrite("referencia2.jpg",img1)
#cv2.imwrite("referencia3.jpg",img2)
#cv2.imwrite("suma1.jpg",img_sum)
#cv2.imwrite("sumaocv.jpg",sumaOCV)
#cv2.imwrite("ponderada1.jpg",mezcla1)
#cv2.imwrite("ponderada2.jpg",mezcla2)

#espera un input y destruye las ventanas
cv2.waitKey(0)
cv2.destroyAllWindows()