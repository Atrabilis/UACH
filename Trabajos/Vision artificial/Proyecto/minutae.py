import cv2

def detectar_puntos_sift(imagen):
    # Inicializar el detector SIFT
    sift = cv2.SIFT_create()

    # Detectar y describir los puntos clave con SIFT
    puntos_clave, descriptores = sift.detectAndCompute(imagen, None)

    # Dibujar los puntos clave en la imagen
    imagen_con_puntos = cv2.drawKeypoints(imagen, puntos_clave, None)
    return imagen_con_puntos

def detectar_esquinas_harris(imagen):
    # Convertir la imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Aplicar el algoritmo de detección de esquinas Harris
    esquinas = cv2.cornerHarris(imagen_gris, 2, 3, 0.04)

    # Marcar las esquinas en la imagen original
    imagen[esquinas > 0.01 * esquinas.max()] = [0, 0, 255]
    return imagen

# Cargar la imagen
imagen = cv2.imread('Dataset/101_1.tif')

# Mostrar los puntos detectados con SIFT
A = detectar_puntos_sift(imagen)
B = detectar_esquinas_harris(imagen)
cv2.imshow('SIFT',A)
cv2.imshow('Harris', B)
cv2.waitKey(0)
cv2.destroyAllWindows()

