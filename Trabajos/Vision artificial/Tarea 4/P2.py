import cv2
import numpy as np

def codificar_cadena(contorno):
    codificacion_cadena = ""

    for pixel in contorno:
        codificacion_cadena += str(pixel)

    return codificacion_cadena

def decodificar_cadena(codificacion_cadena):
    contorno_decodificado = []

    for i in range(len(codificacion_cadena)):
        contorno_decodificado.append(int(codificacion_cadena[i]))

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

# Codificar el contorno utilizando cadena
contorno = np.concatenate(contornos).ravel().tolist()
codificacion_cadena = codificar_cadena(contorno)

# Decodificar el contorno
contorno_decodificado = decodificar_cadena(codificacion_cadena)

# Mostrar la imagen con los contornos
cv2.imshow("Contornos", imagen_contornos)
cv2.imwrite("cadena.jpg",imagen_contornos)
cv2.waitKey(0)
cv2.destroyAllWindows()
