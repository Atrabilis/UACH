import numpy as np
import cv2

img = cv2.imread("imagen1.jpg", cv2.IMREAD_GRAYSCALE)

filaDeCeros =np.zeros((1,img.shape[1]), dtype=np.uint8)
ampliada = np.insert(img, np.s_[::1], filaDeCeros, axis=0)
ampliada = np.vstack((ampliada, filaDeCeros))
columnaDeCeros = np.zeros((ampliada.shape[0],1), dtype=np.uint8)
ampliada = np.insert(ampliada,np.s_[0::1],columnaDeCeros, axis = 1)
ampliada = np.hstack((ampliada,columnaDeCeros))

cv2.imshow('Imagen', img)
cv2.imshow('Ampliada antes de proceso', ampliada)
cv2.waitKey(0)
cv2.destroyAllWindows()


mascara_conv = np.array([[0.25,0.5 ,0.25],[0.5,1 ,0.5],[0.25,0.5 ,0.25]])
region_conv = ampliada[1:-1,1:-1]
sum = 0
sumas = []
indices = []
for i, j in np.ndindex(region_conv.shape[:2]):
    elemento = region_conv[i, j]
    for indice, valor in np.ndenumerate(ampliada[i:i+3,j:j+3]):
        sum += valor*mascara_conv[indice[0],indice[1]]
    sumas.append(sum)
    if len(indices) != region_conv.shape[0]*region_conv.shape[1]:
        indices.append((i,j))
    sum = 0

for i,j in enumerate(indices):
    region_conv[j[0],j[1]] = sumas[i]

cv2.imshow('Imagen', img)
cv2.imshow("Ampliada Despues de la convolucion", region_conv)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Fin del algoritmo")