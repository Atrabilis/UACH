import cv2 as cv

#Lee imagen original
img = cv.imread(r'C:\Users\gschl\Documents\Uach\I Semestre 2022\ELEP233\Tareas\Archivos Python\Imagen\Star.jpg')
#Transforma a escala de grises
imgGris = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#Transforma a imagen binaria
ret,Abin = cv.threshold(imgGris,127,255,0)
area = 0

#Calcula área
for fil in range (img.shape[0]):
    for col in range (img.shape[1]):
        if Abin[fil,col] == 0: area = area + 1

print(area)

# Visualiza imagen binaria
cv.imshow("Imagen Binaria", Abin)
cv.waitKey(0)