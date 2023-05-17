#importa librerias
import numpy as np
import cv2

def operador_kirsch(imagen):
    #Define las máscaras de Kirsch para las 8 direcciones
    mascaras = [
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),  # k0 (norte)
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),  # k1 (noreste)
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),  # k2 (este)
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),  # k3 (sureste)
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),  # k4 (sur)
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),  # k5 (suroeste)
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),  # k6 (oeste)
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])   # k7 (noroeste)
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