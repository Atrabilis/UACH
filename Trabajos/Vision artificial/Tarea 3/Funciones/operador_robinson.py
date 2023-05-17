#importa librerias
import numpy as np
import cv2

def operador_robinson(imagen):
    #Define las máscaras de Kirsch para las 8 direcciones
    mascaras = [
        np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),  # r0 (norte)
        np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]),  # r1 (noreste)
        np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),  # r2 (este)
        np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]]),  # r3 (sureste)
        -1*np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),  # r4 (sur)
        -1*np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]),  # r5 (suroeste)
        -1*np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),  # r6 (oeste)
        -1*np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]])   # r7 (noroeste)
    ]

    #Convierte la imagen a escala de grises si no lo está
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    #Inicializa una matriz para almacenar los resultados
    resultado = np.zeros_like(imagen, dtype=np.uint8)

    #Aplica las máscaras de Kirsch y obtener el máximo en cada píxel
    for mascara in mascaras:
        imagen_filtrada = cv2.filter2D(imagen, -1, mascara)
        resultado = np.maximum(resultado, imagen_filtrada)

    return resultado