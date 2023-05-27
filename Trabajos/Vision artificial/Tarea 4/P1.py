#Importa librerias
import numpy as np
import cv2
import os
from Funciones.clear import clear
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#limpia la consola
clear()

# Cargar la imagen
img = cv2.imread(os.path.dirname(__file__) + "./Codigo de ayuda/figuras.jpg", 0)

#Coeficientes y polinomios

#Cuadrado
coef1 = [322]
x1 = np.linspace(53,176)
poli1 = np.poly1d(coef1)
y1 = poli1(x1)
 
coef2 = [-1.80, 417]
x2 = np.linspace(53,114)
poli2 = np.poly1d(coef2)
y2 = poli2(x2)

coef3 = [1.8,5]
x3 = np.linspace(167-53,176)
poli3 = np.poly1d(coef3)
y3 = poli3(x3)

#rectangulo
coef4 = [-.523,98]
x4 = np.linspace(13,158)
poli4 = np.poly1d(coef4)
y4 = poli4(x4)

coef5 = [-.523,207]
x5 = np.linspace(57,57+158-13)
poli5 = np.poly1d(coef5)
y5 = poli5(x5)

coef6 = [1.93,67]
x6 = np.linspace(13,13+44)
poli6 = np.poly1d(coef6)
y6 = poli6(x6)

coef7 = [1.93,-291]
x7 = np.linspace(158,158+44)
poli7 = np.poly1d(coef7)
y7 = poli7(x7)

#Cuadrado diagonal
coef8 = [.5,6]
x8 = np.linspace(379,379+86)
poli8 = np.poly1d(coef8)
y8 = poli8(x8)

coef9 = [.5,114]
x9 = np.linspace(336,336+86)
poli9 = np.poly1d(coef9)
y9 = poli9(x9)

coef10 = [-1.99,950]
x10 = np.linspace(336,336+43)
poli10 = np.poly1d(coef10)
y10 = poli10(x10)

coef11 = [-1.99,1165]
x11 = np.linspace(422,422+43)
poli11 = np.poly1d(coef11)
y11 = poli11(x11)

#Elipse
a = 75
b = 35
x_center = 402
y_center = 96
angle_degrees = 38  # Ángulo de rotación en grados

# Definir la matriz de transformación afín para la rotación
angle_radians = np.deg2rad(angle_degrees)
rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians)],
                            [np.sin(angle_radians), np.cos(angle_radians)]])

# Generar los puntos en el sistema de coordenadas original
theta = np.linspace(0, 2*np.pi, 100)
x_original = x_center + a * np.cos(theta)
y_original = y_center + b * np.sin(theta)

# Aplicar la transformación de rotación a los puntos
points = np.array([x_original, y_original])
rotated_points = np.dot(rotation_matrix, points - np.array([[x_center], [y_center]]))
x_rotated, y_rotated = rotated_points[0, :] + x_center, rotated_points[1, :] + y_center

# Crear la figura y los ejes
fig, ax = plt.subplots()


# Mostrar la imagen en los ejes
ax.imshow(img,cmap='gray')

# Graficar una curva roja
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
ax.plot(x_rotated, y_rotated, 'r')

# Ajustar los límites de los ejes
ax.set_xlim(0, img.shape[1])
ax.set_ylim(img.shape[0], 0)

# Mostrar el gráfico resultante
plt.show()

#cv2.imshow("img",img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

