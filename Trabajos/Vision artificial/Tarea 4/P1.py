#Este codigo en python codifica cada una de las figuras 
#en figuras.jpg utilizando codigo spline

#Importa librerías
import numpy as np
import cv2
import os
from Funciones.clear import clear
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#Limpia la consola
clear()

#Carga la imagen
img = cv2.imread(os.path.dirname(__file__) + "./Codigo de ayuda/figuras.jpg", 0)

#Coeficientes y polinomios

#Triángulo
coef1 = [322]
x1 = np.linspace(53, 176)
poli1 = np.poly1d(coef1)
y1 = poli1(x1)

coef2 = [-1.80, 417]
x2 = np.linspace(53, 114)
poli2 = np.poly1d(coef2)
y2 = poli2(x2)

coef3 = [1.8, 5]
x3 = np.linspace(167 - 53, 176)
poli3 = np.poly1d(coef3)
y3 = poli3(x3)

parametros_triangulo = np.array([[0, coef1[0], x1[0], x1[-1]],
                                 [coef2[0], coef2[1], x2[0], x2[-1]],
                                 [coef3[0], coef3[1], x3[0], x3[-1]]])

print(parametros_triangulo)

#Rectángulo
coef4 = [-.523, 98]
x4 = np.linspace(13, 158)
poli4 = np.poly1d(coef4)
y4 = poli4(x4)

coef5 = [-.523, 207]
x5 = np.linspace(57, 57 + 158 - 13)
poli5 = np.poly1d(coef5)
y5 = poli5(x5)

coef6 = [1.93, 67]
x6 = np.linspace(13, 13 + 44)
poli6 = np.poly1d(coef6)
y6 = poli6(x6)

coef7 = [1.93, -291]
x7 = np.linspace(158, 158 + 44)
poli7 = np.poly1d(coef7)
y7 = poli7(x7)

parametros_rectangulo = np.array([[coef4[0], coef4[1], int(x4[0]), int(x4[-1])],
                                  [coef5[0], coef5[1], int(x5[0]), int(x5[-1])],
                                  [coef6[0], coef6[1], int(x6[0]), int(x6[-1])],
                                  [coef7[0], coef7[1], int(x7[0]), int(x7[-1])]])

print(parametros_rectangulo)

#Cuadrado diagonal
coef8 = [.5, 6]
x8 = np.linspace(379, 379 + 86)
poli8 = np.poly1d(coef8)
y8 = poli8(x8)

coef9 = [.5, 114]
x9 = np.linspace(336, 336 + 86)
poli9 = np.poly1d(coef9)
y9 = poli9(x9)

coef10 = [-1.99, 950]
x10 = np.linspace(336, 336 + 43)
poli10 = np.poly1d(coef10)
y10 = poli10(x10)

coef11 = [-1.99, 1165]
x11 = np.linspace(422, 422 + 43)
poli11 = np.poly1d(coef11)
y11 = poli11(x11)

parametros_cuadrado = np.array([[coef8[0], coef8[1], int(x8[0]), int(x8[-1])],
                                [coef9[0], coef9[1], int(x9[0]), int(x9[-1])],
                                [coef10[0], coef10[1], int(x10[0]), int(x10[-1])],
                                [coef11[0], coef11[1], int(x11[0]), int(x11[-1])]])

print(parametros_cuadrado)

#Elipse
a = 75
b = 35
x_centro = 402
y_centro = 96
angulo_grados = 38  # Ángulo de rotación en grados

#Define la matriz de transformación afín para la rotación
angulo_radianes = np.deg2rad(angulo_grados)
matriz_rotacion = np.array([[np.cos(angulo_radianes), -np.sin(angulo_radianes)],
                            [np.sin(angulo_radianes), np.cos(angulo_radianes)]])

#Genera los puntos en el sistema de coordenadas original
theta = np.linspace(0, 2 * np.pi, 100)
x_original = x_centro + a * np.cos(theta)
y_original = y_centro + b * np.sin(theta)

#Aplica la transformación de rotación a los puntos
puntos = np.array([x_original, y_original])
puntos_rotados = np.dot(matriz_rotacion, puntos - np.array([[x_centro], [y_centro]]))
x_rotados, y_rotados = puntos_rotados[0, :] + x_centro, puntos_rotados[1, :] + y_centro

#Crea la figura y los ejes
fig, ax = plt.subplots()

#Muestra la imagen en los ejes
ax.imshow(img, cmap='gray')

#Grafica curvas rojas
ax.plot(x1, y1, 'r')
ax.plot(x2, y2, 'r')
ax.plot(x3, y3, 'r')
ax.plot(x4, y4, 'r')
ax.plot(x5, y5, 'r')
ax.plot(x6, y6, 'r')
ax.plot(x7, y7, 'r')
ax.plot(x8, y8, 'r')
ax.plot(x9, y9, 'r')
ax.plot(x10, y10, 'r')
ax.plot(x11, y11, 'r')
ax.plot(x_rotados, y_rotados, 'r')

#Ajusta los límites de los ejes
ax.set_xlim(0, img.shape[1])
ax.set_ylim(img.shape[0], 0)

#Muestra el gráfico resultante
plt.show()



