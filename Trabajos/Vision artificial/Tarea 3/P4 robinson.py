#Este programa implementa el operador de Robinson en Python

#Importe de librerias
import cv2 
import numpy as np
import os
from Funciones.clear import clear
from Funciones.operador_robinson import operador_robinson

#limpia la consola
clear()

#Lee y almacena la imagen
img1 = cv2.imread(os.path.dirname(__file__) + "./imagen1.jpg")
img2 = cv2.imread(os.path.dirname(__file__) + "./imagen2.jpg")
img3 = cv2.imread(os.path.dirname(__file__) + "./Codigo de ayuda/papagayo.jpg")
img4 = cv2.imread(os.path.dirname(__file__) + "./Codigo de ayuda/objetos.jpg")
img5 = cv2.imread(os.path.dirname(__file__) + "./Codigo de ayuda/Star.jpg")

#Aplica el operador Robinson
robinson1= operador_robinson(img1)
robinson2= operador_robinson(img2)
robinson3= operador_robinson(img3)
robinson4= operador_robinson(img4)
robinson5= operador_robinson(img5)

#Muestra y guarda las imagenes, espera un input y destruye las ventanas
cv2.imshow("Operador robinson 1", robinson1)
cv2.imshow("Operador robinson 2", robinson2)
cv2.imshow("Operador robinson 3", robinson3)
cv2.imshow("Operador robinson 4", robinson4)
cv2.imshow("Operador robinson 5", robinson5)
cv2.imwrite("robinson1.jpg", robinson1)
cv2.imwrite("robinson2.jpg", robinson2)
cv2.imwrite("robinson3.jpg", robinson3)
cv2.imwrite("robinson4.jpg", robinson4)
cv2.imwrite("robinson5.jpg", robinson5)
cv2.waitKey(0)
cv2.destroyAllWindows()