import cv2
import numpy as np

img = cv2.imread("imagen1.jpg",cv2.IMREAD_GRAYSCALE)
region_conv1 = np.copy(img[1:-1,1:-1])
region_conv2 = np.copy(img[1:-1,1:-1])

#Filtro Promedio
promedios = []
indices = []
for i, j in np.ndindex(region_conv1.shape[:2]):
    promedios.append(np.mean(img[i:i+3,j:j+3]))
    if len(indices) != region_conv1.shape[0]*region_conv1.shape[1]:
        indices.append((i,j))

for i,j in enumerate(indices):
    region_conv1[j[0],j[1]] = promedios[i]
    
#Filtro Medio
medianas = []
indices = []
for i, j in np.ndindex(region_conv2.shape[:2]):
    matriz_orden = img[i:i+3,j:j+3]
    matriz_orden = np.sort(matriz_orden)
    medianas.append(matriz_orden[1][1])
    if len(indices) != region_conv1.shape[0]*region_conv1.shape[1]:
        indices.append((i,j))

for i,j in enumerate(indices):
    region_conv2[j[0],j[1]] = medianas[i]


cv2.imshow('Imagen', img)
cv2.imshow("Filtro promedio", region_conv1)
cv2.imshow("Filtro Medio", region_conv2)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Programa finalizado")
    