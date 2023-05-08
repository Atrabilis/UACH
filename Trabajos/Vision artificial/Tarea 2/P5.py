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

NUMERO_PIXELES = img.shape[0]*img.shape[1]
niveles_de_gris = [0 for i in range(256)]

for i in img:
    for j in i:
        niveles_de_gris[j] +=1

histograma_deseado_1 = crear_lista_enteros_positivos_aleatorios(NUMERO_PIXELES,niveles_de_gris)
histograma_deseado_2 = crear_lista_enteros_positivos_aleatorios(NUMERO_PIXELES,niveles_de_gris)
histograma_deseado_3 = crear_lista_enteros_positivos_aleatorios(NUMERO_PIXELES,niveles_de_gris)
#print(sum(histograma_deseado_1))
#print(sum(histograma_deseado_2))
#print(sum(histograma_deseado_3))
#print(sum(niveles_de_gris))

print("Programa finalizado")