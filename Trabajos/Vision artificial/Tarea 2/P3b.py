import numpy as np
import cv2

img1 = cv2.imread("imagen1.jpg")
img2 = cv2.imread("imagen2.jpg")

resta1 = img1 - img2
resta2 = img2 - img1

restaOCV1 = cv2.subtract(img1,img2)
restaOCV2 = cv2.subtract(img2,img1)

cv2.imshow("Imagen 1", img1)
cv2.imshow("Imagen 2", img1)
cv2.imshow("resta 1", resta1)
cv2.imshow("resta 2", resta2)
cv2.imshow("resta con cv2.substract() img1 - img2", restaOCV1)
cv2.imshow("resta con cv2.substract() img2 - img1", restaOCV2)

cv2.waitKey(0)
cv2.destroyAllWindows()