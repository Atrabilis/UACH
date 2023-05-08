import numpy as np
import cv2

img1 = cv2.imread("imagen1.jpg")
CONSTANTE = 1

dividida = img1 * (1/CONSTANTE)
divididaOCV = cv2.divide(img1,CONSTANTE)

cv2.imshow("División normal", dividida)
cv2.imshow("División mediante cv2.divide()", divididaOCV)
cv2.waitKey(0)
cv2.destroyAllWindows()