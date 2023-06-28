import cv2
import os

os.system("cls")

def aplicar_filtro_gaussiano(imagen, kernel_size, desviacion):
    # Aplicar filtro gaussiano a la imagen
    imagen_filtrada = cv2.GaussianBlur(imagen, (kernel_size, kernel_size), desviacion)
    return imagen_filtrada

def aplicar_filtro_media(imagen, kernel_size):
    # Aplicar filtro de media a la imagen
    imagen_filtrada = cv2.blur(imagen, (kernel_size, kernel_size))
    return imagen_filtrada

def aplicar_filtro_bilateral(imagen, diametro, sigma_color, sigma_space):
    # Aplicar filtro bilateral a la imagen
    imagen_filtrada = cv2.bilateralFilter(imagen, diametro, sigma_color, sigma_space)
    return imagen_filtrada

# Cargar la imagen de huella dactilar
imagen_huella = cv2.imread("Dataset/101_1.tif", 0)  # Asegúrate de especificar la ruta correcta de tu imagen

# Parámetros de los filtros
tamaño_kernel_gaussiano = 5  # Tamaño del kernel para el filtro gaussiano (debe ser un número impar)
desviacion_gaussiano = 0  # Desviación estándar del filtro gaussiano

tamaño_kernel_media = 5  # Tamaño del kernel para el filtro de media (debe ser un número impar)

diametro_bilateral = 9  # Diámetro del filtro bilateral
sigma_color_bilateral = 75  # Valor sigma para el componente de color en el filtro bilateral
sigma_space_bilateral = 75  # Valor sigma para el componente espacial en el filtro bilateral

# Aplicar los filtros a la imagen de huella dactilar
imagen_filtrada_gaussiano = aplicar_filtro_gaussiano(imagen_huella, tamaño_kernel_gaussiano, desviacion_gaussiano)
imagen_filtrada_media = aplicar_filtro_media(imagen_huella, tamaño_kernel_media)
imagen_filtrada_bilateral = aplicar_filtro_bilateral(imagen_huella, diametro_bilateral, sigma_color_bilateral, sigma_space_bilateral)

# Mostrar la imagen original y los resultados de los filtros
cv2.imshow("Imagen original", imagen_huella)
cv2.imshow("Imagen filtrada (Gaussiano)", imagen_filtrada_gaussiano)
cv2.imshow("Imagen filtrada (Media)", imagen_filtrada_media)
cv2.imshow("Imagen filtrada (Bilateral)", imagen_filtrada_bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()
