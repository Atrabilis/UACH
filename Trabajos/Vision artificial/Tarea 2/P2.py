import numpy as np
import cv2

#Función que amplia la imagen con filas y columnas de ceros.
def ampliar(img):
    filaDeCeros =np.zeros((1,img.shape[1]), dtype=np.uint8)
    ampliada = np.insert(img, np.s_[::1], filaDeCeros, axis=0)
    ampliada = np.vstack((ampliada, filaDeCeros))
    columnaDeCeros = np.zeros((ampliada.shape[0],1), dtype=np.uint8)
    ampliada = np.insert(ampliada,np.s_[0::1],columnaDeCeros, axis = 1)
    ampliada = np.hstack((ampliada,columnaDeCeros))
    return ampliada

#Función que realiza el zoom de la imagen.
def realizar_zoom(img,factor):
    iteracion_zoom = 1
    copia_local = np.copy(img)  
    while iteracion_zoom <= factor:
        sumas = []
        indices = []
        ampliada = ampliar(copia_local)
        sum = 0
        region_conv = ampliada[1:-1,1:-1]
        for i, j in np.ndindex(region_conv.shape[:2]):
            for indice, valor in np.ndenumerate(ampliada[i:i+3,j:j+3]):
                sum += valor*mascara_conv[indice[0],indice[1]]
            sumas.append(sum)
            if len(indices) != region_conv.shape[0]*region_conv.shape[1]:
                indices.append((i,j))
            sum = 0

        for i,j in enumerate(indices):
            region_conv[j[0],j[1]] = sumas[i]
        iteracion_zoom+=1
        copia_local = region_conv
    return copia_local

#lee y almacena la imagen
img = cv2.imread("imagen1.jpg", cv2.IMREAD_GRAYSCALE)
#cuantas veces se desea ejecutar el algoritmo
factor = int(input("Ingrese cuantas veces desea realizar el zoom: "))
#mascara de convolucion utilizada
mascara_conv = np.array([[0.25,0.5 ,0.25],[0.5,1 ,0.5],[0.25,0.5 ,0.25]])

#imagen con el zoom implementado
zoom= realizar_zoom(img,factor)


cv2.imshow('Imagen', img)
cv2.imshow("Ampliada Despues de la convolucion", zoom)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Fin del algoritmo")