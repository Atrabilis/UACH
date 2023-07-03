import numpy as np
import cv2 as cv
import os

# Lee imagen
img = cv.imread(os.path.dirname(__file__) + '\Star.jpg')

# Transforma a escala de grises
imgGris = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Especifica valor de umbral
umbral = 127

# Transforma a imagen binaria
ret, imgBin = cv.threshold(imgGris,umbral,255,0)

# Encuentra puntos de contorno
contours,hierarchy = cv.findContours(imgBin, 1, 2)
cnt = contours[0]

# Calcula centroide
M = cv.moments(cnt)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
print("Centroide", cx,cy)

# Calcula perímetro
perimeter = cv.arcLength(cnt,True)
print("perimetro", perimeter)

# Visualiza imagen binaria
cv.imshow("Imagen Binaria", imgBin)
cv.waitKey(0)
