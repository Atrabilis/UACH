import cv2
import numpy as np
import random

def crear_lista_enteros_positivos_aleatorios(M,niveles_de_gris):
    lista = [0] * 256
    
    for i in range(256):
        if sum(lista) < M-max(niveles_de_gris):
            lista[i] = random.randint(0,max(niveles_de_gris))
        else:
            break
    
    suma_actual = sum(lista)
    
    while suma_actual < M:
        i = random.randint(0, 255)
        lista[i] += 1
        suma_actual += 1
    
    return lista

img = cv2.imread("imagen1.jpg",cv2.IMREAD_GRAYSCALE)

#Histograma original
NUMERO_PIXELES = img.shape[0]*img.shape[1]
niveles_de_gris = [0 for i in range(256)]

for i in img:
    for j in i:
        niveles_de_gris[j] +=1
        


#Histogramas deseados aleatorios (Paso II)
histograma_deseado_1 = crear_lista_enteros_positivos_aleatorios(NUMERO_PIXELES,niveles_de_gris)
histograma_deseado_2 = crear_lista_enteros_positivos_aleatorios(NUMERO_PIXELES,niveles_de_gris)
histograma_deseado_3 = crear_lista_enteros_positivos_aleatorios(NUMERO_PIXELES,niveles_de_gris)


#Mapeos (paso III)
mapeo1= [0] * 256
mapeo2= [0] * 256
mapeo3= [0] * 256

for i in range(1,256):
    mapeo1[i] = round((sum(histograma_deseado_1[:i])/sum(histograma_deseado_1))*255)
    mapeo2[i] = round((sum(histograma_deseado_2[:i])/sum(histograma_deseado_2))*255)
    mapeo3[i] = round((sum(histograma_deseado_3[:i])/sum(histograma_deseado_3))*255)
    
print("Programa finalizado")