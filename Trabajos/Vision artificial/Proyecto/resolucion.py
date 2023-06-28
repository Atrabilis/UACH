import cv2
import numpy as np

def reducir_resolucion_decimacion(imagen, factor_reduccion):
    # Obtener las dimensiones originales de la imagen
    altura_original, ancho_original = imagen.shape[:2]

    # Calcular las nuevas dimensiones de la imagen después de la reducción
    nueva_altura = altura_original // factor_reduccion
    nueva_ancho = ancho_original // factor_reduccion

    # Realizar decimación mediante indexación
    imagen_reducida = imagen[::factor_reduccion, ::factor_reduccion]

    return imagen_reducida

# Cargar la imagen de huella dactilar
imagen_huella = cv2.imread("Dataset/101_1.tif")

# Especificar el factor de reducción deseado (por ejemplo, 3 para reducir a un tercio)
factor_reduccion = 3

# Reducir la resolución de la imagen utilizando decimación
imagen_reducida = reducir_resolucion_decimacion(imagen_huella, factor_reduccion)

# Mostrar la imagen original y la imagen reducida
cv2.imshow("Imagen original", imagen_huella)
cv2.imshow("Imagen reducida", imagen_reducida)
cv2.waitKey(0)
cv2.destroyAllWindows()