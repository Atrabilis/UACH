import cv2
import numpy as np

def codificar_rle(contorno):
    contorno_codificado = []
    pixel_actual = contorno[0]
    conteo = 1

    for i in range(1, len(contorno)):
        if contorno[i] == pixel_actual:
            conteo += 1
        else:
            contorno_codificado.append((pixel_actual, conteo))
            pixel_actual = contorno[i]
            conteo = 1

    contorno_codificado.append((pixel_actual, conteo))
    return contorno_codificado

def decodificar_rle(contorno_codificado):
    contorno_decodificado = []

    for pixel, conteo in contorno_codificado:
        contorno_decodificado.extend([pixel] * conteo)

    return contorno_decodificado

# Cargar imagen de OpenCV
imagen = cv2.imread('./Codigo_de_ayuda/objetos.jpg', cv2.IMREAD_GRAYSCALE)

# Aplicar umbralización de Otsu para obtener una imagen binaria
_, umbral_otsu = cv2.threshold(imagen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Obtener contornos de los objetos en la imagen binaria
contornos, _ = cv2.findContours(umbral_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Crear una copia de la imagen en color para dibujar los contornos
imagen_contornos = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)

# Dibujar los contornos en la imagen
cv2.drawContours(imagen_contornos, contornos, -1, (0, 255, 0), 2)

# Codificar el contorno utilizando RLE
contorno = np.concatenate(contornos).ravel().tolist()
contorno_codificado = codificar_rle(contorno)

# Decodificar el contorno
contorno_decodificado = decodificar_rle(contorno_codificado)

# Mostrar la imagen con los contornos
cv2.imshow("Contornos", imagen_contornos)
cv2.imwrite("runlength.jpg", imagen_contornos)
cv2.waitKey(0)
cv2.destroyAllWindows()
