""" 
Este programa realiza un acercamiento
digital de una imagen RGB, almacenada en la variable img, mediante el método
de convolución de máscara. El programa solicita como dato de entrada el
factor del zoom (un número entero entre 1 y 10.) y realiza el zoom digital. Luego compara con la función resize de OpenCV
""" 
#se importan librerias
import numpy as np
import cv2
import os

#Borra terminal en windows
os.system('cls')

#Función que amplia la imagen con filas y columnas de ceros.
def ampliar(img):
    #se insertan filas y columnas de cero intercalados
    filaDeCeros =np.zeros((1,img.shape[1]), dtype=np.uint8)
    ampliada = np.insert(img, np.s_[::1], filaDeCeros, axis=0)
    ampliada = np.vstack((ampliada, filaDeCeros))
    columnaDeCeros = np.zeros((ampliada.shape[0],1), dtype=np.uint8)
    ampliada = np.insert(ampliada,np.s_[0::1],columnaDeCeros, axis = 1)
    ampliada = np.hstack((ampliada,columnaDeCeros))
    return ampliada

#Función que realiza el zoom de la imagen.
print("Comenzando algoritmo")
def realizar_zoom(img,factor):
    iteracion_zoom = 1
    copia_local = np.copy(img)  
    while iteracion_zoom <= factor:
        print("inicio de iteracion" , iteracion_zoom)
        sumas = []
        indices = []
        ampliada = ampliar(copia_local)
        sum = 0
        region_conv = ampliada[1:-1,1:-1]
        for i, j in np.ndindex(region_conv.shape[:2]):
            for indice, valor in np.ndenumerate(ampliada[i:i+3,j:j+3]):
                sum += valor*mascara_conv[indice[0],indice[1]]
            sumas.append(sum)
            if len(indices) != region_conv.shape[0]*region_conv.shape[1]:
                indices.append((i,j))
            sum = 0

        for i,j in enumerate(indices):
            region_conv[j[0],j[1]] = sumas[i]
        iteracion_zoom+=1
        copia_local = region_conv
        print("fin de iteracion ",iteracion_zoom-1)
    return copia_local



#cuantas veces se desea ejecutar el algoritmo
factor = int(input("Ingrese cuantas veces desea realizar el zoom: "))

#Matriz que almacena las coordenadas del click del mouse.
point_matrix = np.zeros((2,2),int)
 
counter = 0
flag = 0

#Funcion que captura el evento de click del mouse
def mousePoints(event,x,y,flags,params):
    global counter
    # Left button mouse click event opencv
    if event == cv2.EVENT_LBUTTONDOWN:
        point_matrix[counter] = x,y
        print("primer vertice en ", (x,y))
        counter = counter + 1
        if counter == 2:
                print("Con la imagen en primer plano, presione una tecla o cierre la imagen")
            
        

#Lee y almacena la imagen
img = cv2.imread("imagen1.jpg", cv2.IMREAD_GRAYSCALE)
#cv2.imwrite("referencia1.jpg", img)

print("Haga click derecho en el vertice de inicio y en el vertice final, de arriba a"+ 
      " la izquierda hacia abajo a la derecha")

while True:
    for x in range (0,2):
        cv2.circle(img,(point_matrix[x][0],point_matrix[x][1]),3,(0,255,0),cv2.FILLED)
 
    if counter == 2:
        #variables que almacenan las coordenadas de los clicks.
        starting_x = point_matrix[0][0]
        starting_y = point_matrix[0][1]
 
        ending_x = point_matrix[1][0]
        ending_y = point_matrix[1][1]
        
        # Dibuja un rectangulo en el area de interes
        cv2.rectangle(img, (starting_x, starting_y), (ending_x, ending_y), (0, 255, 0), 3)
        print("presione una tecla en una de las imagenes para comenzar las iteraciones")
 
        #recorta la imagen
        img_cropped = img[starting_y:ending_y, starting_x:ending_x]
 
    # Muestra imagen original
    cv2.imshow("Imagen original", img)
    cv2.setWindowProperty("Imagen original", cv2.WND_PROP_TOPMOST, 1)
    # Evento de click en la imagen original
    cv2.setMouseCallback("Imagen original", mousePoints)

    # refresca la ventana
    cv2.waitKey(0)

    if flag == 1: break
    if counter == 2: 
        flag += 1

#mascara de convolucion utilizada
mascara_conv = np.array([[0.25,0.5 ,0.25],[0.5,1 ,0.5],[0.25,0.5 ,0.25]])

#imagen con el zoom implementado
zoom = realizar_zoom(img_cropped,factor)
print(type(zoom))

#Muestra las imagenes
cv2.imshow("Imagen Despues del zoom digital", zoom)

# zoom de cv
def zoom(img, zoom_factor):
    return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)

zoomed_and_cropped = zoom(img_cropped, factor)

#cv2.imwrite("zoomcv1.jpg", zoomed_and_cropped)

cv2.imshow("Imagen zoom cv", zoomed_and_cropped )

print("presione cualquier tecla para salir") 
#Espera un input y destruye las ventanas
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Fin del algoritmo")