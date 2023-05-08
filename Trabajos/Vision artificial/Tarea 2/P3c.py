import numpy as np
import cv2

img1 = cv2.imread("imagen1.jpg")
CONSTANTE = 1

multiplicada = img1*CONSTANTE

multiplicadaOCV = cv2.multiply(img1, CONSTANTE)

cv2.imshow("Multiplicacion normal", multiplicada)
cv2.imshow("Multiplicacion mediante cv2.multiply()", multiplicadaOCV)

cv2.waitKey(0)
cv2.destroyAllWindows()