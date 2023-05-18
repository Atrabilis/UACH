#Este programa implementa el operador Prewitt en Python

#Importe de librerias
import cv2 
import numpy as np
import os
from Funciones.clear import clear
from Funciones.operador_prewitt import operador_prewitt

#limpia la consola
clear()

#Lee y almacena la imagen
img1 = cv2.imread(os.path.dirname(__file__) + "./imagen1.jpg")
img2 = cv2.imread(os.path.dirname(__file__) + "./imagen2.jpg")
img3 = cv2.imread(os.path.dirname(__file__) + "./Codigo de ayuda/papagayo.jpg")
img4 = cv2.imread(os.path.dirname(__file__) + "./Codigo de ayuda/objetos.jpg")
img5 = cv2.imread(os.path.dirname(__file__) + "./Codigo de ayuda/Star.jpg")

#Aplica el operador prewitt
prewitt1= operador_prewitt(img1)
prewitt2= operador_prewitt(img2)
prewitt3= operador_prewitt(img3)
prewitt4= operador_prewitt(img4)
prewitt5= operador_prewitt(img5)

#Muestra y guarda las imagenes, espera un input y destruye las ventanas
cv2.imshow("Operador Prewitt 1", prewitt1)
cv2.imshow("Operador Prewitt 2", prewitt2)
cv2.imshow("Operador Prewitt 3", prewitt3)
cv2.imshow("Operador Prewitt 4", prewitt4)
cv2.imshow("Operador Prewitt 5", prewitt5)
cv2.imwrite("prewitt1.jpg", prewitt1)
cv2.imwrite("prewitt2.jpg", prewitt2)
cv2.imwrite("prewitt3.jpg", prewitt3)
cv2.imwrite("prewitt4.jpg", prewitt4)
cv2.imwrite("prewitt5.jpg", prewitt5)

cv2.waitKey(0)
cv2.destroyAllWindows()