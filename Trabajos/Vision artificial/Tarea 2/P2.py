#Se importan librerias
import os
import cv2 
import numpy as np 

#Borra terminal en windows
os.system('cls')

# Pregunta a usuario por el factor del zoom
factor = int(input("Ingrese el factor del zoom a aplicar en la imagen:"))

# Create point matrix get coordinates of mouse click on image
point_matrix = np.zeros((2,2),int)

img = cv2.imread(os.path.dirname(__file__) + '\imagen1.jpg')

cv2.imshow("asdfasdf", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
