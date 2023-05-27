
import cv2
import numpy as np

def seguimiento_contorno(imagen_binaria):
    # Obtener las dimensiones de la imagen
    alto, ancho = imagen_binaria.shape

    # Encontrar el punto inicial en la primera fila de la imagen
    punto_inicial = None
    for y in range(alto):
        for x in range(ancho):
            if imagen_binaria[y, x] == 0:
                punto_inicial = (y, x)
                break
        if punto_inicial is not None:
            break

    if punto_inicial is None:
        return None

    # Inicializar la lista de puntos del contorno
    contorno = [punto_inicial]

    # Definir las direcciones de vecindad (8 vecinos)
    direcciones_vecindad = [(0, -1), (-1, -1), (-1, 0), (-1, 1),
                            (0, 1), (1, 1), (1, 0), (1, -1)]

    # Iniciar el seguimiento del contorno
    punto_actual = punto_inicial
    direccion_actual = 0

    while True:
        # Calcular la siguiente posición en la dirección actual
        direccion_actual = (direccion_actual + 6) % 8
        vecino_actual = (punto_actual[0] + direcciones_vecindad[direccion_actual][0],
                         punto_actual[1] + direcciones_vecindad[direccion_actual][1])

        # Verificar si el vecino está dentro de los límites de la imagen
        if 0 <= vecino_actual[0] < alto and 0 <= vecino_actual[1] < ancho:
            # Verificar si el vecino pertenece al contorno
            if imagen_binaria[vecino_actual[0], vecino_actual[1]] == 0:
                # Agregar el vecino al contorno
                contorno.append(vecino_actual)
                punto_actual = vecino_actual
                direccion_actual = (direccion_actual + 4) % 8
                continue

        # Verificar si se ha completado el contorno
        if punto_actual == punto_inicial and direccion_actual == 0:
            break

    return contorno
