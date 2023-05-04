#Este progra transforma la imagen RGB a una imagen HSI,
#luego elimina la información referente a tonalidad y saturación para obtener una
#imagen en escala de grises.

#Parametros: imagen: la imagen a utilizar en forma de string.
#            write: si se asigna True guarda la imagen manipulada en el directorio de trabajo.
#            show: si se asigna True, muestra las imagenes producidas
#            onlyShow: si se asigna False, retorna la imagen modificada
def grises(imagen, write = False, show = False, onlyShow = True):
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
    
    #Se convierte la imagen BGR a HSV
    originalHSV = cv2.cvtColor(original, cv2.COLOR_RGB2HSV)
    #Se extrae solo el canal de valor o intensidad
    originalGrises = originalHSV[:,:,2]
    #Se muestran las distintas imagenes
    
    if show:
        cv2.imshow("Original", original)
        cv2.imshow("HSV", originalHSV)
        cv2.imshow("Grises", originalGrises)
        
        #Espera un input del usuario durante n milisegundos, 0 indica indefinidamente
        n = 0
        cv2.waitKey(n)
        #Elimina las ventanas abiertas
        cv2.destroyAllWindows
    
    if write:
        cv2.imwrite("original_grises.jpg",originalGrises)
    
    if onlyShow:
        return originalGrises
    
grises("original.jpg",show = True, write=True)
