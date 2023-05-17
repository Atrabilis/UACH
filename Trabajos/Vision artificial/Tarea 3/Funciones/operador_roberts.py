import numpy as np

def operador_roberts(img):
    #Se crea una copia de la imagen de entrada
    I = np.copy(img)
    rows, cols = img.shape

    #Se itera la matriz
    for fil in range(1, rows):
        for col in range(1, cols):
            # Se aplica la fórmula del operador Roberts
            pixel = abs(img[fil, col] - img[fil-1, col-1]) or abs(img[fil, col-1] - img[fil-1, col])
            I[fil, col] = pixel

    return I