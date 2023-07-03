#Este programa reemplaza los colores de una imagen arbitraria con los colores de la tabla siguiente.
#En el color original se considera que 1 significa >= 120 y 0 significa
#<120. En color nuevo 1 significa 255 y 0 significa 0.
"""
| Color Original RGB | Color Nuevo RGB |
|--------------------|-----------------|
| 0 0 0              | 1 1 1           |
| 0 0 1              | 0 1 0           |
| 0 1 0              | 1 1 0           |
| 0 1 1              | 1 0 0           |
| 1 0 0              | 0 1 1           |
| 1 0 1              | 0 0 1           |
| 1 1 0              | 1 0 1           |
| 1 1 1              | 0 0 0           |
"""

#Parametros: imagen: la imagen a utilizar en forma de string.
#            write: si se asigna True guarda la imagen manipulada en el directorio de trabajo.
#            show: si se asigna True, muestra las imagenes producidas
#            onlyShow: si se asigna False, retorna la imagen modificada
def cambiar_colores(imagen = None, write = False, show=False, onlyShow = True):
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
            
    #Convierte la imagen de BGR a RGB 
    originalRGB = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    #Se crea una matriz logica para contener los valores que cumplen la condicion 
    #(>= 120 indica un 1 y <120 indica un cero) con el fin de hacer el codigo mas legible
    matriz_logica = originalRGB>=120

    #Se itera en las filas y columnas de la matriz de imagen 
    for fila in range(originalRGB.shape[0]):
        for columna in range(originalRGB.shape[1]):
            #En este nivel el color del pixel es interpretado como un array lógico,
            #color original almacena el pixel de la forma en que está en la tabla del enunciado
            #(1,0,1), (0,0,0), etc
            color_original = matriz_logica[fila,columna,:]
            #Se comprueban las distintas condiciones del enunciado y se modifica
            #el color segun corresponda 
            if not color_original[0] and not color_original[1] and not color_original[2]:
                color_nuevo = np.array([255,255,255])
                originalRGB[fila,columna,:] = color_nuevo 
            elif not color_original[0] and not color_original[1] and  color_original[2]:
                color_nuevo = np.array([0,255,0])
                originalRGB[fila,columna,:] = color_nuevo
            elif not color_original[0] and color_original[1] and not color_original[2]:
                color_nuevo = np.array([255,255,0])
                originalRGB[fila,columna,:] = color_nuevo
            elif not color_original[0] and color_original[1] and  color_original[2]:
                color_nuevo = np.array([255,0,0])
                originalRGB[fila,columna,:] = color_nuevo
            elif color_original[0] and not color_original[1] and not color_original[2]:
                color_nuevo = np.array([0,255,255])
                originalRGB[fila,columna,:] = color_nuevo
            elif  color_original[0] and not color_original[1] and  color_original[2]:
                color_nuevo = np.array([0,0,255])
                originalRGB[fila,columna,:] = color_nuevo 
            elif color_original[0] and  color_original[1] and not color_original[2]:
                color_nuevo = np.array([255,0,255])
                originalRGB[fila,columna,:] = color_nuevo
            else: 
                color_nuevo = np.array([0,0,0])
                originalRGB[fila,columna,:] = color_nuevo
    
    if show:
        #Muestra la imagen original
        cv2.imshow("original",original)
        #Muestra la imagen modificada
        cv2.imshow("modificada",originalRGB)
        #Espera un input del usuario durante n milisegundos, 0 indica indefinidamente
        n = 0
        cv2.waitKey(n)
        #Elimina las ventanas abiertas
        cv2.destroyAllWindows
        
    if write:
        cv2.imwrite("original_threshold.jpg",originalRGB)
    
    if not onlyShow:
        return originalRGB
    
cambiar_colores("original.jpg", write = True, show = True)

        

        
