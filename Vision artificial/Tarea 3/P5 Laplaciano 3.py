#Este programa implementa el primer operador Laplaciano visto en clases en Python.

#Importe de librerias
import cv2 
import numpy as np
import os
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
laplaciano1= operador_laplaciano(img1, 3)
laplaciano2= operador_laplaciano(img2, 3)
laplaciano3= operador_laplaciano(img3, 3)
laplaciano4= operador_laplaciano(img4, 3)
laplaciano5= operador_laplaciano(img5, 3)


#Muestra y guarda las imagenes, espera un input y destruye las ventanas
cv2.imshow("Operador Laplaciano 1", laplaciano1)
cv2.imshow("Operador Laplaciano 2", laplaciano2)
cv2.imshow("Operador Laplaciano 3", laplaciano3)
cv2.imshow("Operador Laplaciano 4", laplaciano4)
cv2.imshow("Operador Laplaciano 5", laplaciano5)
#cv2.imwrite("laplaciano31.jpg", laplaciano1)
#cv2.imwrite("laplaciano32.jpg", laplaciano2)
#cv2.imwrite("laplaciano33.jpg", laplaciano3)
#cv2.imwrite("laplaciano34.jpg", laplaciano4)
#cv2.imwrite("laplaciano35.jpg", laplaciano5)
cv2.waitKey(0)
cv2.destroyAllWindows()