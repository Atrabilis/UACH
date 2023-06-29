import cv2

def procesamiento(imagen):
    # Conversión a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Redimensionamiento de la imagen a la mitad del tamaño original
    imagen_redimensionada = cv2.resize(imagen_gris, None, fx=0.5, fy=0.5)
    
    return imagen_redimensionada

imagen_original = cv2.imread('Dataset/101_1.tif')  # Lee la imagen de huella dactilar
imagen_procesada = procesamiento(imagen_original)  # Aplica el procesamiento de conversión a escala de grises

# Muestra la imagen original y la imagen procesada
cv2.imshow('Imagen Original', imagen_original)
cv2.imshow('Imagen Procesada', imagen_procesada)
cv2.waitKey(0)
cv2.destroyAllWindows()
