"""este programa implementa el filtro promedio y el filtro medio y luego los aplica a una imagen
contaminada con ruido “sal y pimienta”. Luego compara el resultado obtenido con la función medianBlur de
OpenCV. """

#importa librerias
import cv2
import numpy as np

#lee la imagen y la almacena
img = cv2.imread("Imagen ruidosa.jpg",cv2.IMREAD_GRAYSCALE)
#define regionnes de filtrado
region_conv1 = np.copy(img[1:-1,1:-1])
region_conv2 = np.copy(img[1:-1,1:-1])
#crea una matriz de ceros para añadir ruido
img_ruido = np.zeros_like(img)
cv2.randn(img_ruido, 0, 50)

#Filtro Promedio
def filtroPromedio(region_conv1):
    promedios = []
    indices = []
    for i, j in np.ndindex(region_conv1.shape[:2]):
        promedios.append(np.mean(img[i:i+3,j:j+3]))
        if len(indices) != region_conv1.shape[0]*region_conv1.shape[1]:
            indices.append((i,j))

    for i,j in enumerate(indices):
        region_conv1[j[0],j[1]] = promedios[i]
    return region_conv1
    
    
#Filtro Medio
def filtroMedio(region_conv2):
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
    return(region_conv2)

#aplica ambos filtros a la imagen de referencia
filtromedio = filtroMedio(region_conv2)
filtropromedio = filtroPromedio(region_conv1)


#añade ruido a la imagen
img_ruidosa = cv2.add(img, img_ruido)

#aplica filtro de mediana de openCV
img_blur = cv2.medianBlur(img, 3)

cv2.imshow('Imagen', img)
cv2.imshow("Filtro promedio", filtropromedio)
cv2.imshow("Filtro Medio", filtromedio)
cv2.imshow("Filtro Medio openCV", img_blur)
#cv2.imshow("Imagen ruidosa", img_ruidosa)

#cv2.imwrite("filtropromedio.jpg",filtropromedio)
#cv2.imwrite("filtromedio.jpg",filtromedio)
#cv2.imwrite("filtromedioocv2.jpg",img_blur)
#cv2.imwrite("Imagen ruidosa.jpg", img_ruidosa)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Programa finalizado")
    