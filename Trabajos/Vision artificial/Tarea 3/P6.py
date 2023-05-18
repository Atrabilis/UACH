import cv2
import os
from Funciones.clear import clear

clear()

#Lee y almacena la imagen
img = cv2.imread(os.path.dirname(__file__) + '\star.jpg')

#Transforma a escala de grises
imgGris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Especifica valor de umbral
umbral = 127

#Transforma a imagen binaria
ret, imgBin = cv2.threshold(imgGris,umbral,255,0)

#Encuentra puntos de contorno
contours,hierarchy = cv2.findContours(imgBin, 1, 2)
cnt = contours[0]

#Cuenta y calcula el perimetro de un objeto
perimetro = 0
for i in range(len(cnt)):
    #Punto actual
    point1 = cnt[i][0]
    #Punto siguiente
    point2 = cnt[(i+1) % len(cnt)][0] #El modulo asegura que el índice i+1 esté 
    #dentro del rango válido de índices del contorno cnt.
    
    #Calculamos la distancia entre los puntos
    distance = cv2.norm(point1, point2, cv2.NORM_L2)
    perimetro += distance

print("El perímetro del objeto con el algoritmo implementado es:", perimetro,"\n")

#Calcula perímetro con openCV
perimetro2 = cv2.arcLength(cnt,True)
print("Perimetro obtenido utilizando cv2.arcLength(): ", perimetro2)
cv2.waitKey(0)

