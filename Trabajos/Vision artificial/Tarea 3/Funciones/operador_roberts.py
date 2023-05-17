#Importa Librerias
import numpy as np

def operador_roberts(img):
    #Crea una copia de la imagen de entrada
    I = np.copy(img)
    rows, cols = img.shape

    #Itera la matriz
    for fil in range(1, rows):
        for col in range(1, cols):
            #Aplica la fórmula del operador Roberts
            pixel = abs(img[fil, col] - img[fil-1, col-1]) or abs(img[fil, col-1] - img[fil-1, col])
            I[fil, col] = pixel

    return I