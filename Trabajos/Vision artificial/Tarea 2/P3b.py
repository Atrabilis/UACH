"""Este programa reliza la resta algebraica de dos imagenes, cuidando de que no haya saturación. 
Luego la compara con la resta obtenida por openCV."""

#se importan librerias
import numpy as np
import cv2

#se lee y almacena las imagenes
img1 = cv2.imread("imagen1.jpg")
img2 = cv2.imread("imagen2.jpg")

#Se limita el valor de los pixeles de las imagenes a 200 como máximo, esto para no saturar la imagen en la resta
img1 = np.clip(img1, 0, 200)
img2 = np.clip(img2, 0, 200)

#se normalizan las imagenes a 255
img1_norm = img1 / 255.0
img2_norm = img2 / 255.0

#se realiza la resta directa
img_res1 = img1_norm - img2_norm
img_res2 = img2_norm - img1_norm

#se vuelve a multiplicar la matriz por 255, limitando sus valores entre 0 y 255 y ademas dejando el formato
#de la matriz en uint8 para ser leida por openCV
img_res1 = np.clip(img_res1 * 255, 0, 255).astype(np.uint8)
img_res2 = np.clip(img_res2 * 255, 0, 255).astype(np.uint8)

#se realiza la resta con openCV
restaOCV1 = cv2.subtract(img1,img2)
restaOCV2 = cv2.subtract(img2,img1)

#se muestran las imagenes
cv2.imshow("Imagen 1", img1)
cv2.imshow("Imagen 2", img2)
cv2.imshow("Resta img1 - img2", img_res1)
cv2.imshow("Resta img2 - img1", img_res2)
cv2.imshow("resta con cv2.substract() img1 - img2", restaOCV1)
cv2.imshow("resta con cv2.substract() img2 - img1", restaOCV2)

#cv2.imwrite("resta1.jpg",img_res1)
#cv2.imwrite("resta2.jpg",img_res2)
#cv2.imwrite("restaocv1.jpg",restaOCV1)
#cv2.imwrite("restaocv2.jpg",restaOCV2)

#espera un input y destruye las ventanas
cv2.waitKey(0)
cv2.destroyAllWindows()