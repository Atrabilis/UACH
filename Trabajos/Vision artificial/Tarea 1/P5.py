#Este programa obtiene el negativo de la imagen RGB y el negativo de la imagen en escala
#de grises de las preguntas anteriores, ademas muestra cada imagen original junto a su respectiva imagen negativa.

#Parametros: imagen: la imagen a utilizar en forma de string.
#            write: si se asigna True guarda la imagen manipulada en el directorio de trabajo.
#            show: si se asigna True, muestra las imagenes producidas
#            onlyShow: si se asigna False, retorna la imagen modificada

def negativo(imagenRGB, imagenGris, show = False, write = False, onlyShow = True):
     #Se importan las librerias a utilizar
    import cv2 
    import numpy as np
    
    # Verifica si la imagen fue leída correctamente
    try:
        #Imagen original, -1 equivalente a cv2.IMREAD_UNCHANGED.
        original = cv2.imread(imagenRGB,-1)
        originalGris = cv2.imread(imagenGris,-1)
    except cv2.error as error:
        print("Error: ", error)
        original = None
        originalGris = None
        
    if original is None or originalGris is None:
        print("\n La imagen no se encontró o el nombre es incorrecto.\n")
    
    #Se obtienen los negativos de las imagenes
    original_negativo = 255 - original
    originalGris_negativo = 255 - originalGris
    print(originalGris_negativo.shape)
    
    if show:
        cv2.imshow("Negativo de la imagen original", original_negativo)
        cv2.imshow("imagen original", original)
        cv2.imshow("imagen en escala de grises", originalGris)
        cv2.imshow("Negativo de la imagen en escala de grises", originalGris_negativo)
        
        #Espera un input del usuario durante n milisegundos, 0 indica indefinidamente
        n = 0
        cv2.waitKey(n)
        #Elimina las ventanas abiertas
        cv2.destroyAllWindows
    
    if write:
        cv2.imwrite("original_negativo.jpg", original_negativo)
        cv2.imwrite("original_grises_negativo.jpg", originalGris_negativo)
    
    if not onlyShow:
        return(original_negativo, originalGris_negativo)


negativo("original.jpg", "original_grises.jpg", show = True, write = True)
    
    
    