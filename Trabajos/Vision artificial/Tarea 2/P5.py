"""Este programa crea una especificación de histograma en Python con el algoritmo visto en clases"""

#importe de librerias
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

#Funcion que crea los histogramas con enteros aleatorios
def crear_lista_enteros_positivos_aleatorios(M,niveles_de_gris):
    lista = [0] * 256
    
    for i in range(256):
        if sum(lista) < M-max(niveles_de_gris)*45:
            lista[i] = random.randint(0,int(max(niveles_de_gris)))
        else:
            break
    
    suma_actual = sum(lista)
    
    while suma_actual < M:
        i = random.randint(0, 255)
        lista[i] += 1
        suma_actual += 1
    
    return lista

#Funcion que ecualiza histogramas
def ecualizacion_histograma(histograma):
    suma_desplazamiento = []

    for index,j in enumerate(histograma):
        suma_desplazamiento.append(sum(histograma[:index+1])/NUMERO_PIXELES)
        suma_desplazamiento[index] = round(suma_desplazamiento[index]*255)
    return(suma_desplazamiento)

img = cv2.imread("imagen1.jpg",cv2.IMREAD_GRAYSCALE)
NUMERO_PIXELES = img.shape[0]*img.shape[1]

#Funcion que retorna histograma de imagen
def histograma(img):
    niveles_de_gris = [0 for i in range(256)]
    for i in img:
        for j in i:
            niveles_de_gris[j] +=1
    return(niveles_de_gris)

#Ontencion del histograma original
niveles_de_gris = histograma(img)

#Ecualización de Histograma (Paso I)
suma_desplazamiento= ecualizacion_histograma(niveles_de_gris)

#Histogramas deseados aleatorios (Paso II)
histograma_deseado= crear_lista_enteros_positivos_aleatorios(NUMERO_PIXELES,niveles_de_gris)


#Mapeo (paso III)
mapeo1= [0] * 256
mapeo2= [0] * 256
mapeo3= [0] * 256

for i in range(1,256):
    mapeo1[i] = round((sum(histograma_deseado[:i])/sum(histograma_deseado))*255)


#se grafican los histogramas
fig, axs = plt.subplots(2, 1)

axs[0].plot(list(range(256)), niveles_de_gris)
axs[0].set_title('Histograma original')

axs[1].plot(list(range(256)), histograma_deseado)
axs[1].set_title('Histograma deseado')


#se ajusta el tamaño de los subplots
plt.tight_layout()

#Se muestra el gráfico
plt.show()
    
print("Programa finalizado")