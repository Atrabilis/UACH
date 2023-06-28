import cv2

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

    # Binarización local adaptativa utilizando el método de Sauvola
    umbral_vecindario = 15  # Tamaño del vecindario para el cálculo local
    constante_sauvola = 0.2  # Constante de ponderación para el umbral adaptativo
    imagen_binarizada_sauvola = cv2.adaptiveThreshold(imagen_reducida, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, umbral_vecindario, constante_sauvola)

    # Binarización local adaptativa utilizando el método de Otsu
    _, imagen_binarizada_otsu = cv2.threshold(imagen_reducida, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Mostrar las imágenes resultantes
    cv2.imshow("Imagen Original", imagen)
    cv2.imshow("Filtro Bilateral", imagen_filtrada)
    cv2.imshow("Reducción de Resolución", imagen_reducida)
    cv2.imshow("Binarización Sauvola", imagen_binarizada_sauvola)
    cv2.imshow("Binarización Otsu", imagen_binarizada_otsu)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Cargar la imagen de huella dactilar
imagen_huella = cv2.imread("Dataset/101_1.tif")

# Aplicar el procesamiento de imagen
procesamiento_de_imagen(imagen_huella)
