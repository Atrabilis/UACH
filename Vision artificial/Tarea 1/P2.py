#Este programa crea un vector en N^3 que indica la cantidad de pixeles que tiene cada nivel de color  de las bandas
#R, G y B

#Parametros: imagen: la imagen a utilizar en forma de string
#            show: si se asigna true grafica la cantidad de niveles de Azul, Rojo y verde en la imagen 
#                  utilizando matplotlib, notar que el metodo show() pausa la ejecucion del código
#                  haste que se cierren los graficos.
#            onlyShow: si se asigna False, retorna el vector
#            export: si se asigna True, se exporta el vector resultante a un archivo.txt
def vectorRGB(imagen = None, show = False, onlyShow = True, export = False):
    
    #Se importan las librerias a utilizar
    import cv2 
    import numpy as np

    # Verifica si la imagen fue leída correctamente
    try:
        #Imagen original, -1 equivalente a cv2.IMREAD_UNCHANGED.
        original = cv2.imread(imagen,-1)
    except cv2.error as error:
        print("Error: ", error)
        original = None
        
    if original is None:
        print("\n La imagen no se encontró o el nombre es incorrecto.\n")
    
    #Matriz de zeros para contar la cantidad de pixeles para cada nivel de color
    niveles_color = np.zeros((3,256), dtype = int)
    
    #Divide la imagen en las tres bandas
    B, G, R = cv2.split(original)

    
    #Se recorre una matriz con las mismas dimensiones de la imagen con propositos de indexacion
    #luego se agrega el nivel de azul, verde y rojo a niveles_color 
    for fila in range(len(B)): # En cada fila
        for columna in range(len(B[fila])): #En cada columna la fila iterada
            niveles_color[0,R[fila, columna]] +=1 #Se suma uno al nivel de rojo correspondiente
            niveles_color[1,G[fila, columna]] +=1 #Se suma uno al nivel de verde correspondiente
            niveles_color[2,B[fila, columna]] +=1 #Se suma uno al nivel de azul correspondiente  
    
    if show:
        from  matplotlib import pyplot as plt
        #Se crea una figura con 3 subplots
        fig, (ax1,ax2,ax3) = plt.subplots(3,1)
        ax1.plot(range(256), niveles_color[0,:], color = "red")
        ax1.set_title("Rojo")
        ax2.plot(range(256), niveles_color[1,:], color = "green")
        ax2.set_title("Verde")
        ax3.plot(range(256), niveles_color[2,:], color = "blue")
        ax3.set_title("Azul")
        #Se asigna un label compartido al eje y
        fig.text(0.01, 0.5, "Cantidad de veces que un nivel de color \n se repite en la imagen", 
                 va="center", 
                 rotation="vertical",
                 multialignment="center")
         #Se asigna un label compartido al eje x
        fig.text(0.5, 0.04, 
                 "Niveles de color", 
                 ha="center")
        fig.subplots_adjust(wspace=0.5, hspace=0.8)
        plt.show()
        
    if export:
        np.savetxt("original_niveles_color.txt", niveles_color,header= "Niveles de color", fmt = "%d") #Formato entero 
         
    if not onlyShow:
        return niveles_color

vector = vectorRGB("original.jpg", show = True, onlyShow= False, export = True)
print("Shape:",vector.shape,"\n")
print("Vector: \n", vector)
input()
    
original[fila,columna,2] #[255,136,245]