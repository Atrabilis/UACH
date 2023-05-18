#Este programa implementa el primer operador Laplaciano visto en clases en Python.

#Importe de librerias
import cv2 
import os
import numpy as np
from Funciones.clear import clear
from Funciones.operador_laplaciano import operador_laplaciano

#limpia la consola
clear()

#Lee y almacena la imagen
img1 = cv2.imread(os.path.dirname(__file__) + "./imagen1.jpg")
img2 = cv2.imread(os.path.dirname(__file__) + "./imagen2.jpg")
img3 = cv2.imread(os.path.dirname(__file__) + "./Codigo de ayuda/papagayo.jpg")
img4 = cv2.imread(os.path.dirname(__file__) + "./Codigo de ayuda/objetos.jpg")
img5 = cv2.imread(os.path.dirname(__file__) + "./Codigo de ayuda/Star.jpg")

#Aplica el operador prewitt
laplaciano1= operador_laplaciano(img1, 1)
laplaciano2= operador_laplaciano(img2, 1)
laplaciano3= operador_laplaciano(img3, 1)
laplaciano4= operador_laplaciano(img4, 1)
laplaciano5= operador_laplaciano(img5, 1)


#Muestra y guarda las imagenes, espera un input y destruye las ventanas
cv2.imshow("Operador Laplaciano 1", laplaciano1)
cv2.imshow("Operador Laplaciano 2", laplaciano2)
cv2.imshow("Operador Laplaciano 3", laplaciano3)
cv2.imshow("Operador Laplaciano 4", laplaciano4)
cv2.imshow("Operador Laplaciano 5", laplaciano5)
cv2.imwrite("laplaciano11.jpg", laplaciano1)
cv2.imwrite("laplaciano12.jpg", laplaciano2)
cv2.imwrite("laplaciano13.jpg", laplaciano3)
cv2.imwrite("laplaciano14.jpg", laplaciano4)
cv2.imwrite("laplaciano15.jpg", laplaciano5)
cv2.waitKey(0)
cv2.destroyAllWindows()