"""Este programa es capaz de recortar una región de una
imagen de la manera descrita en clases. El programa solicita como dato
de entrada los números de fila y columna correspondientes a los vértices
superior izquierdo e inferior derecho de la región a recortar. Además, el
programa autocorrege la posición de la región de recorte en caso de que
el usuario la posicione fuera de los márgenes de la imagen o no ingrese los
vértices en el orden solicitado. El programa almacena la imagen a recortar en la variable
img."""

#Se importan las librerias
import cv2
import numpy as np
import os

 
#Lee la imagen
img = cv2.imread(os.path.dirname(__file__) + '\imagen1.jpg')
#Coordenadas de los vértices
f1 = int(input("Ingrese la fila del vértice superior izquierdo: "))
c1 = int(input("Ingrese la columna del vértice superior izquierdo: "))
f2 = int(input("Ingrese la fila del vértice inferior derecho: "))
c2 = int(input("Ingrese la columna del vértice inferior derecho: "))

vertices = [f1,c1,f2,c2]
# Comprueba que los datos hayan sido ingresados correctamente

#esta sección de código asegura que minimamente se genere una imagen de 1x1 
for indice, vertice in enumerate(vertices):
    if vertice < 0:
        if indice == 2 or indice == 3: 
            vertices[indice] = 1
        else:  vertices[indice] = 0
f1,c1,f2,c2 = vertices

if c1 > img.shape[1] or c2 > img.shape[1]: c1,c2 = (img.shape[1]-1,img.shape[1])
if f1 > img.shape[0] or f2 > img.shape[0]: f1,f2 = (img.shape[0]-1,img.shape[0])

img_recortada = img[f1:f2,c1:c2,:]

#Muestra las imagenes
cv2.imshow('Imagen de entrada', img)
cv2.imshow('Imagen recortada', img_recortada)

# Esperar a que el usuario presione una tecla
cv2.waitKey(0)

# Cierra las ventanas
cv2.destroyAllWindows()




