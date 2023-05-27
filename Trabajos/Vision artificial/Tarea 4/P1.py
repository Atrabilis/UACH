#Importa librerias
import numpy as np
import cv2
import os

#Lee y almacena la imagen
img = cv2.imread(os.path.dirname(__file__) + "./Codigo de ayuda/figuras.jpg", 0)



cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
