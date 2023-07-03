#importa librerias
import cv2
import numpy as np

def operador_prewitt(imagen):
    #Convierte la imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    #Aplica el filtro Prewitt en las direcciones X e Y
    prewitt_y = cv2.filter2D(imagen_gris, -1, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
    prewitt_x = cv2.filter2D(imagen_gris, -1, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))
    
    #Combina las respuestas en la imagen resultante
    prewitt_combinado = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)
    
    return prewitt_combinado