#Este programa implementa el operador de Kirsch en Python

#Importe de librerias
import cv2 
import os
import numpy as np
from Funciones.clear import clear
from Funciones.operador_kirsch import operador_kirsch


#limpia la consola
clear()

#Lee y almacena la imagen
img1 = cv2.imread(os.path.dirname(__file__) + "./imagen1.jpg")
img2 = cv2.imread(os.path.dirname(__file__) + "./imagen2.jpg")
img3 = cv2.imread(os.path.dirname(__file__) + "./Codigo de ayuda/papagayo.jpg")
img4 = cv2.imread(os.path.dirname(__file__) + "./Codigo de ayuda/objetos.jpg")
img5 = cv2.imread(os.path.dirname(__file__) + "./Codigo de ayuda/Star.jpg")

#Aplica el operador kirsch
kirsch1= operador_kirsch(img1)
kirsch2= operador_kirsch(img2)
kirsch3= operador_kirsch(img3)
kirsch4= operador_kirsch(img4)
kirsch5= operador_kirsch(img5)

#Muestra y guarda las imagenes, espera un input y destruye las ventanas
cv2.imshow("Operador kirsch 1", kirsch1)
cv2.imshow("Operador kirsch 2", kirsch2)
cv2.imshow("Operador kirsch 3", kirsch3)
cv2.imshow("Operador kirsch 4", kirsch4)
cv2.imshow("Operador kirsch 5", kirsch5)
cv2.imwrite("kirsch1.jpg", kirsch1)
cv2.imwrite("kirsch2.jpg", kirsch2)
cv2.imwrite("kirsch3.jpg", kirsch3)
cv2.imwrite("kirsch4.jpg", kirsch4)
cv2.imwrite("kirsch5.jpg", kirsch5)

cv2.waitKey(0)
cv2.destroyAllWindows()