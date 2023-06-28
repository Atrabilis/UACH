import cv2
import numpy as np

def procesamiento_de_imagen(imagen):
    # Convertir la imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Parámetros del filtro bilateral
    diametro_bilateral = 9  # Diámetro del filtro bilateral
    sigma_color_bilateral = 75  # Valor sigma para el componente de color en el filtro bilateral
    sigma_space_bilateral = 75  # Valor sigma para el componente espacial en el filtro bilateral

    # Aplicar filtro bilateral a la imagen en escala de grises
    imagen_filtrada = cv2.bilateralFilter(imagen_gris, diametro_bilateral, sigma_color_bilateral, sigma_space_bilateral)

    # Reducción de resolución utilizando decimación
    factor_reduccion = 3
    imagen_reducida = imagen_filtrada[::factor_reduccion, ::factor_reduccion]

    # Binarización utilizando el método de Otsu
    _, imagen_binarizada = cv2.threshold(imagen_reducida, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Detección de minutiae
    puntos_bifurcacion, puntos_terminacion = detectar_minutiae(imagen_binarizada)

    return imagen_binarizada, puntos_bifurcacion, puntos_terminacion


def detectar_minutiae(imagen_binaria):
    # Crear un kernel de 3x3 para el operador de Hessiano
    kernel = np.array([[1, 1, 1],
                       [1, -8, 1],
                       [1, 1, 1]], dtype=np.float32)

    # Aplicar el operador de Hessiano para detectar los puntos de bifurcación y terminación
    hessiano = cv2.filter2D(imagen_binaria, -1, kernel)

    # Encontrar los puntos de bifurcación y terminación
    puntos_bifurcacion = np.argwhere(hessiano < 0)
    puntos_terminacion = np.argwhere(hessiano > 0)

    return puntos_bifurcacion, puntos_terminacion


# Cargar la imagen de huella dactilar
imagen_huella = cv2.imread("Dataset/101_1.tif")

# Aplicar el procesamiento de imagen y la detección de minutiae
imagen_binarizada, puntos_bifurcacion, puntos_terminacion = procesamiento_de_imagen(imagen_huella)

# Mostrar la imagen binarizada
cv2.imshow("Imagen Binarizada", imagen_binarizada)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Imprimir los puntos de bifurcación y terminación
print("Puntos de bifurcación:", puntos_bifurcacion)
print("Puntos de terminación:", puntos_terminacion)
