#Este codigo codifica cada uno de los objetos de la imagen “objetos.jpg” utilizando
#código Run Length. Luego crea una nueva imagen en donde los objetos sean generados
#a partir del código cadena obtenido anteriormente.
import cv2
import numpy as np

def codificar_rle(contorno):
    #Inicializa la lista para almacenar el contorno codificado en RLE
    contorno_codificado = []
    #Inicializa el pixel actual y el conteo
    pixel_actual = contorno[0]
    conteo = 1

    for i in range(1, len(contorno)):
        #Comprueba si el pixel actual es igual al siguiente
        if contorno[i] == pixel_actual:
            #Incrementa el conteo
            conteo += 1
        else:
            #Agrega el par (pixel, conteo) al contorno codificado
            contorno_codificado.append((pixel_actual, conteo))
            #Actualiza el pixel actual y reinicia el conteo
            pixel_actual = contorno[i]
            conteo = 1

    #Agrega el último par (pixel, conteo) al contorno codificado
    contorno_codificado.append((pixel_actual, conteo))
    return contorno_codificado

def decodificar_rle(contorno_codificado):
    #Inicializa la lista para almacenar el contorno decodificado
    contorno_decodificado = []

    for pixel, conteo in contorno_codificado:
        #Agrega el pixel repetido según el conteo al contorno decodificado
        contorno_decodificado.extend([pixel] * conteo)

    return contorno_decodificado

#Carga la imagen de OpenCV
imagen = cv2.imread('./Codigo_de_ayuda/objetos.jpg', cv2.IMREAD_GRAYSCALE)

#Aplica umbralización de Otsu para obtener una imagen binaria
_, umbral_otsu = cv2.threshold(imagen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#Obtiene los contornos de los objetos en la imagen binaria
contornos, _ = cv2.findContours(umbral_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#Crea una copia de la imagen en color para dibujar los contornos
imagen_contornos = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)

#Dibuja los contornos en la imagen
cv2.drawContours(imagen_contornos, contornos, -1, (0, 255, 0), 2)

#Codifica el contorno utilizando RLE
contorno = np.concatenate(contornos).ravel().tolist()
contorno_codificado = codificar_rle(contorno)

#Decodifica el contorno
contorno_decodificado = decodificar_rle(contorno_codificado)

#Muestra la imagen con los contornos
cv2.imshow("Contornos", imagen_contornos)
cv2.imwrite("runlength.jpg", imagen_contornos)
cv2.waitKey(0)
cv2.destroyAllWindows()
