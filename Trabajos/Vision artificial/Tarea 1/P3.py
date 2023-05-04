#Este programa expresa una imagen arbitraria en en escala de rojos, verde, azul, cyan, magenta y
#amarillo.


#Parametros: imagen: la imagen a utilizar en forma de string.
#            write: si se asigna True guarda la imagen manipulada en el directorio de trabajo.
#            show: si se asigna True, muestra las imagenes producidas
#            onlyShow: si se asigna False, retorna la imagen modificada
def colores_en_distintas_escalas(imagen = None, write = False, show = False, onlyShow = True):
    #Se importan las librerias a utilizar
    import cv2 
    import numpy as np
    import matplotlib as plt
    # Verifica si la imagen fue leída correctamente
    try:
        #Imagen original, -1 equivalente a cv2.IMREAD_UNCHANGED.
        original = cv2.imread(imagen,-1)
    except cv2.error as error:
        print("Error: ", error)
        original = None
        
    if original is None:
        print("\n La imagen no se encontró o el nombre es incorrecto.\n")
    
    #Se crean matrices de ceros con la misma forma que la imagen original,
    #luego se extraen las escalas de colores correspondientes de la imagen original
    #para asignarselas a las matrices de ceros
    escala_rojos = np.zeros_like(original)
    escala_verdes = np.zeros_like(original)
    escala_azules = np.zeros_like(original)
    escala_cyan = np.zeros_like(original)
    escala_magenta = np.zeros_like(original)
    escala_amarillo = np.zeros_like(original)
    
    escala_rojos[:,:,2] = original[:,:,2] #Se extrae la capa roja
    escala_verdes[:,:,1] = original[:,:,1] #Se extrae la capa verde
    escala_azules[:,:,0] = original[:,:,0] #Se extrae la capa azul
    escala_cyan[:,:,0:2] = original[:,:,0:2] #Se extrae la capa azul y verde
    escala_magenta[:,:,0:3:2] = original[:,:,0:3:2] #Se extrae la azul y roja
    escala_amarillo[:,:,1:3] = original[:,:,1:3] #Se extrae la verde y roja
    
    #Se muestran las distintas imagenes
    if show:
        cv2.imshow("original", original)
        cv2.imshow("Escala de rojo", escala_rojos)
        cv2.imshow("Escala de azul", escala_azules)
        cv2.imshow("Escala de verde", escala_verdes)
        cv2.imshow("Escala de cyan", escala_cyan)
        cv2.imshow("Escala de magenta", escala_magenta)
        cv2.imshow("Escala de amarillo", escala_amarillo)
        
        
        #Espera un input del usuario durante n milisegundos, 0 indica indefinidamente
        n = 0
        cv2.waitKey(n)
        #Elimina las ventanas abiertas
        cv2.destroyAllWindows
            
    if write:
        cv2.imwrite("original_escala_de_rojo.jpg", escala_rojos)
        cv2.imwrite("original_escala_de_azul.jpg", escala_azules)
        cv2.imwrite("original_escala_de_verde.jpg", escala_verdes)
        cv2.imwrite("original_escala_de_cyan.jpg", escala_cyan)
        cv2.imwrite("original_escala_de_magenta.jpg", escala_magenta)
        cv2.imwrite("original_escala_de_amarillo.jpg", escala_amarillo)
    

    if not onlyShow:
        return (escala_rojos,
                escala_verdes,
                escala_azules,
                escala_cyan,
                escala_magenta,
                escala_amarillo)

colores_en_distintas_escalas("original.jpg", write = True, show = True)
    

