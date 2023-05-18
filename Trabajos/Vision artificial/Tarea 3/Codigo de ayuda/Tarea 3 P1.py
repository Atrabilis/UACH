import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

# Lee imagen
img = cv.imread(os.path.dirname(__file__) + '\papagayo.jpg')


# Transforma a escala de grises
img_gris = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Calcula histograma
histogram = cv.calcHist([img],[0],None,[256],[0,256])

# Histograma
x = np.arange(0, 256)
y = histogram
# coeficientes del polinomio
pol = np.polyfit(x,y,12)  

xx = np.linspace(min(x),max(x))
# valor del polinomio en los puntos de la matriz xx
yy = np.polyval(pol,xx)  

# Aproximación polinomial de histograma
hist_app_poly = np.full(256, 0, dtype = float)

for i in range(256):
    c = 0
    for k in range(13):
         c = c + pol[12 - k]*i**k
    hist_app_poly[i] = c

# Previene que la aproximación polinomial del histograma tenga valores negativos    
hist_app_poly_min = min(hist_app_poly)
if hist_app_poly_min < 0: hist_app_poly = hist_app_poly- hist_app_poly_min

# Búsqueda de mínimos locales
m = np.r_[True, hist_app_poly[1:] < hist_app_poly[:-1]] & np.r_[hist_app_poly[:-1] < hist_app_poly[1:], True]
# índices de mínimos locales
m_list = [i for i, xd in enumerate(m) if xd]

# Búsqueda de máximos locales
M = np.r_[True, hist_app_poly[1:] > hist_app_poly[:-1]] & np.r_[hist_app_poly[:-1] > hist_app_poly[1:], True]
# índices de máxmos locales
M_list = [i for i, xd in enumerate(M) if xd]

# Matriz con índice y valor de mínimos locales
m_array = np.array(m_list)
m_index = m_array.reshape(len(m_array),1)
m_matriz = np.concatenate((m_index, hist_app_poly[m_index]), axis = 1)

# Matriz con índice y valor de mínimos locales
M_array = np.array(M_list)
M_index = M_array.reshape(len(M_array),1)
M_matriz = np.concatenate((M_index, hist_app_poly[M_index]), axis = 1)

# Cálculo de 2 máximos locales mayores
M_max = M_matriz.copy()
indexMax1 = np.argmax(M_max[:,1])
M_max[indexMax1,1] = 0
indexMax2 = np.argmax(M_max[:,1])

# Índices de 2 máximos mayores
indexMax1 = int(M_max[indexMax1,0])
indexMax2 = int(M_max[indexMax2,0])
indexMaxA = min(indexMax1, indexMax2)
indexMaxB = max(indexMax1, indexMax2)

filter_arr = []
filter_values = []

# Cálculo de mínimo menor ubicado entre 2 máximos mayores (umbral)
min_value = max(hist_app_poly[M_index])
umbral = 0

for element in m_index:
    if element > indexMaxA and element < indexMaxB and hist_app_poly[element] < min_value:
            min_value = hist_app_poly[element]
            umbral = element
            
img_bin = img_gris.copy()

# Genera imagen binaria a partir de imagen en escala de grises
for fil in range(img.shape[0]):
    for col in range(img.shape[1]):
        if img_gris[fil, col] > umbral: img_bin[fil, col] = 255
        else: img_bin[fil, col] = 0

# Despliega imágenes
cv.imshow("Imagen escala de gris", img_gris)
cv.imshow("Imagen binaria", img_bin)

# Gráfico de histograma - aproximación polinomial - umbral
d = 1
plt.axis([min(xx)-d, max(xx)+d, min(yy)-d, max(yy)+max(yy)*0.04])
plt.plot(x, hist_app_poly, '-', x, y, 'bo', indexMaxA, hist_app_poly[indexMaxA], 'ro', indexMaxB, hist_app_poly[indexMaxB], 'ro', umbral, hist_app_poly[umbral], 'go')
plt.show()