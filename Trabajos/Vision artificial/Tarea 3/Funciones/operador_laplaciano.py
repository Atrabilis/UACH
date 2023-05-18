#importa librerias
import numpy as np
import cv2

def operador_laplaciano(imagen,mascara):
    # Definir las máscaras laplacianas
    mascaras = [np.array([[0, -1, 0],
                        [-1, 4, -1],
                        [0, -1, 0]]),
                np.array([[1, -2, 1],
                        [-2, 4, -2],
                        [1, -2, 1]]),
                np.array([[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]])
                ]
    # Aplicar la convolución
    if mascara == 1:
        laplaciano = cv2.filter2D(imagen, -1, mascaras[0])
    elif mascara == 2:
        laplaciano = cv2.filter2D(imagen, -1, mascaras[1])
    elif mascara == 3:
        laplaciano = cv2.filter2D(imagen, -1, mascaras[2])
    else: 
        print("la mascara debe ser un numero entre 1 y 3")
        

    return laplaciano