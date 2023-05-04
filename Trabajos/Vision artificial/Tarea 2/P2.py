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

#Máscara de convolución utilizada en el proceso de zoom
mascara_conv = np.array([[0.25,0.5 ,0.25],[0.5,1 ,0.5],[0.25,0.5 ,0.25]])

#proceso para crear la matriz ampliada con ceros
filaDeCeros =np.zeros((1,img.shape[1],img.shape[2]), dtype = img.dtype)
ampliada = np.insert(img, np.s_[::1], filaDeCeros, axis=0)
ampliada = np.vstack((ampliada, filaDeCeros))
columnaDeCeros = np.zeros((ampliada.shape[0],1,img.shape[2]), dtype = ampliada.dtype)
ampliada = np.insert(ampliada,np.s_[0::1],columnaDeCeros, axis = 1)
ampliada = np.hstack((ampliada,columnaDeCeros))


cv2.imshow("asdfasdf", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
