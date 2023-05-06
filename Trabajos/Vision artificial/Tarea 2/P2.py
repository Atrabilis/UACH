for var in globals().copy():
    if not var.startswith("_"):
        del globals()[var]
import numpy as np
import cv2

"""img = np.array([
    [[1],[2],[3]],
    [[4],[5],[6]],
    [[7],[8],[9]]
])"""
# Cargar imagen en color
img = cv2.imread("imagen1.jpg")

print(img.shape,"\n")
filaDeCeros =np.zeros((1,img.shape[1],img.shape[2]), dtype=np.uint8)
#print(filaDeCeros)
#print(filaDeCeros.shape)
ampliada = np.insert(img, np.s_[::1], filaDeCeros, axis=0)
#print(ampliada.shape,"\n")
ampliada = np.vstack((ampliada, filaDeCeros))
#print(ampliada.shape,"\n")
columnaDeCeros = np.zeros((ampliada.shape[0],1,img.shape[2]), dtype=np.uint8)
#print(columnaDeCeros)
#print(columnaDeCeros.shape)
ampliada = np.insert(ampliada,np.s_[0::1],columnaDeCeros, axis = 1)
#print(ampliada.shape,"\n")
ampliada = np.hstack((ampliada,columnaDeCeros))
print(ampliada.shape,"\n")
#print(ampliada,"\n")

cv2.imshow('Imagen', img)
cv2.imshow('Ampliada antes de proceso', ampliada)
cv2.waitKey(0)
cv2.destroyAllWindows()

mascara_conv = np.array([[0.25,0.5 ,0.25],[0.5,1 ,0.5],[0.25,0.5 ,0.25]])
region_conv = ampliada[1:-1,1:-1,:]
#print(mascara_conv.shape)
#print(region_conv.shape)
#print(img.shape)
#print("Región de convolución","\n \n",region_conv, "\n")
sum = 0
sumas = np.zeros((region_conv.shape[0], region_conv.shape[1]))
indices = []
for i, j in np.ndindex(region_conv.shape[:2]):
    #print((i,j))
    elemento = region_conv[i, j]
    print("Elemento en la posición ({},{}) es: {}".format(i, j, elemento))
    print("elementos adyacentes")
    print(ampliada[i:i+3,j:j+3])
    for indice, valor in np.ndenumerate(ampliada[i:i+3,j:j+3]):
        print("indice: ",indice[:2], "valor: ",valor)
        print("valor mascara convolución: " ,mascara_conv[indice[0],indice[1]])
        print(valor*mascara_conv[indice[0],indice[1]])
        sum += valor*mascara_conv[indice[0],indice[1]]
        
    sumas.append(sum)
    if len(indices) != region_conv.shape[0]*region_conv.shape[1]:
        #print("agregando indices")
        indices.append((i,j))
        #print(len(indices))
        #print("indices = ",indices)
    #print("sumas = ",sumas)
    #print("sum : ", sum)
    #print("ampliada: ", ampliada)
    sum = 0
#print(indices)
#print(len(indices))
#print("sumas:","\n",sumas)
#print(len(sumas))
for i,j in enumerate(indices):
    #print(i,j)
    ampliada[j[0],j[1]] = sumas[i] 
#print("img:","\n",img)
#print("resultado:","\n",ampliada)


# Mostrar imagen en una ventana
cv2.imshow('Imagen', img)
cv2.imshow('Ampliada', ampliada)
cv2.waitKey(0)
cv2.destroyAllWindows()

