import numpy as np
import cv2

img1 = cv2.imread("imagen1.jpg")
img2 = cv2.imread("imagen2.jpg")

suma = img1 + img2
sumaOCV = cv2.add(img1,img2)
mezcla1 = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)
mezcla2 = cv2.addWeighted(img1, 0.3, img2, 0.7, 0)

cv2.imshow("Imagen 1", img1)
cv2.imshow("Imagen 2", img2)
cv2.imshow("Suma", suma)
cv2.imshow("Suma con cv2.add()", sumaOCV)
cv2.imshow("Suma Ponderada img1 dominante", mezcla1)
cv2.imshow("Suma Ponderada img2 dominante", mezcla2)
cv2.waitKey(0)
cv2.destroyAllWindows()