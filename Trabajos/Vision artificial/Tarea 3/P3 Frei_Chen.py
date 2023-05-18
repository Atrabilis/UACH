#Este programa implementa el operador Frei-Chen en Python

#Importe de librerias
import cv2 
import numpy as np
import os
from Funciones.clear import clear
from Funciones.operador_frei_chen import operador_frei_chen

#limpia la consola
clear()

#Lee y almacena la imagen
img1 = cv2.imread(os.path.dirname(__file__) + "./imagen1.jpg")
img2 = cv2.imread(os.path.dirname(__file__) + "./imagen2.jpg")
img3 = cv2.imread(os.path.dirname(__file__) + "./Codigo de ayuda/papagayo.jpg")
img4 = cv2.imread(os.path.dirname(__file__) + "./Codigo de ayuda/objetos.jpg")
img5 = cv2.imread(os.path.dirname(__file__) + "./Codigo de ayuda/Star.jpg")

#Aplica el operador Frei Chen
frei_chen1= operador_frei_chen(img1)
frei_chen2= operador_frei_chen(img2)
frei_chen3= operador_frei_chen(img3)
frei_chen4= operador_frei_chen(img4)
frei_chen5= operador_frei_chen(img5)

#Muestra y guarda las imagenes, espera un input y destruye las ventanas
cv2.imshow("Operador Frei Chen 1", frei_chen1)
cv2.imshow("Operador Frei Chen 2", frei_chen2)
cv2.imshow("Operador Frei Chen 3", frei_chen3)
cv2.imshow("Operador Frei Chen 4", frei_chen4)
cv2.imshow("Operador Frei Chen 5", frei_chen5)
#cv2.imwrite("freichen1.jpg", frei_chen1)
#cv2.imwrite("freichen2.jpg", frei_chen2)
#cv2.imwrite("freichen3.jpg", frei_chen3)
#cv2.imwrite("freichen4.jpg", frei_chen4)
#cv2.imwrite("freichen5.jpg", frei_chen5)
cv2.waitKey(0)
cv2.destroyAllWindows()