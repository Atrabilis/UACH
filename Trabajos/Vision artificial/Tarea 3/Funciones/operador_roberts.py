import numpy as np

def operador_roberts(img):
#Se crea una copia de la imagen de entrada
    I = np.copy(img)
#Se itera la matriz
    for fil, fila in enumerate(img):
        for col, elemento in enumerate(fila):
#Se aplica la suma de las diferencias de los píxeles vecinos diagonales
            I[fil,col] = 255 if abs(I[fil,col] - I[fil-1,col-1]) or abs(I[fil,col-1] - I[fil-1,col]) else 0
    return I