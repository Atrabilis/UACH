import cv2
import numpy as np

img = cv2.imread("imagen1.jpg",cv2.IMREAD_GRAYSCALE)

h, w = img.shape[:2]
M = h * w

region_conv = np.copy(img[1:-1,1:-1])
sum = 0
promedios = []
indices = []
for i, j in np.ndindex(region_conv.shape[:2]):
    promedios.append(np.mean(img[i:i+3,j:j+3]))
    if len(indices) != region_conv.shape[0]*region_conv.shape[1]:
        indices.append((i,j))
    sum = 0

for i,j in enumerate(indices):
    region_conv[j[0],j[1]] = promedios[i]
    
cv2.imshow('Imagen', img)
cv2.imshow("imagen despues del filtro", region_conv)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Programa finalizado")
    