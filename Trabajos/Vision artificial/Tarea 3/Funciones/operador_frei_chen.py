#import alibrerias
import cv2
import numpy as np

def operador_frei_chen(imagen):
    #Converte la imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    #Define los kernels del operador Frei-Chen
    kernel_x = np.array([[-1, -np.sqrt(2), -1], [0, 0, 0], [1, np.sqrt(2), 1]])
    kernel_y = np.array([[-1, 0, 1], [-np.sqrt(2), 0, np.sqrt(2)], [-1, 0, 1]])

    #Aplicarel filtro Frei-Chen en las direcciones X y Y
    freichen_x = cv2.filter2D(imagen_gris, -1, kernel_x)
    freichen_y = cv2.filter2D(imagen_gris, -1, kernel_y)

    #Calcula el módulo del gradiente
    gradiente = np.hypot(freichen_x, freichen_y)

    #Normaliza el gradiente a escala de 0 a 255
    gradiente = (gradiente / np.max(gradiente)) * 255
    gradiente = np.uint8(gradiente)

    return gradiente
