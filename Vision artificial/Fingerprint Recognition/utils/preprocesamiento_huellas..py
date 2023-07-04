import cv2
import csv
import numpy as np
import os
from sklearn.model_selection import train_test_split

os.system("cls")

def extract_minutiae(image_path, save_images=False):
    # Cargar la imagen en escala de grises
    image = cv2.imread(image_path, 0)

    # Aplicar ecualización del histograma
    image_equalized = cv2.equalizeHist(image)

    # Aplicar un filtro de suavizado para reducir el ruido
    image_blurred = cv2.GaussianBlur(image_equalized, (5, 5), 0)

    # Aplicar umbralización adaptativa para binarizar la imagen
    _, image_thresholded = cv2.threshold(image_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Aplicar operaciones morfológicas para mejorar la imagen binaria
    kernel = np.ones((3, 3), np.uint8)
    image_eroded = cv2.erode(image_thresholded, kernel, iterations=2)
    image_dilated = cv2.dilate(image_eroded, kernel, iterations=1)

    # Encontrar los contornos de las crestas de la huella
    contours, _ = cv2.findContours(image_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar los contornos en la imagen original
    image_with_contours = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)

    # Extraer las características de las crestas (minutias)
    minutiae = []
    for contour in contours:
        hull = cv2.convexHull(contour, clockwise=False, returnPoints=True)
        for i in range(1, len(hull) - 1, 5):
            start = tuple(hull[i-1][0])
            end = tuple(hull[i+1][0])
            minutiae.append((start, end))

    if save_images:
        # Crear la carpeta "contornos" si no existe
        if not os.path.exists('contornos'):
            os.makedirs('contornos')

        # Guardar la imagen con los contornos en la carpeta "contornos"
        contours_image_path = os.path.join('contornos', os.path.basename(image_path).replace(".tif",".jpg"))
        cv2.imwrite(contours_image_path, image_with_contours)

        # Crear la carpeta "minutias" si no existe
        if not os.path.exists('minutias'):
            os.makedirs('minutias')

        # Dibujar las minutias sobre la imagen original
        image_with_minutiae = image.copy()
        for (start, end) in minutiae:
            cv2.circle(image_with_minutiae, start, 5, (0, 0, 255), 2)
            cv2.line(image_with_minutiae, start, end, (0, 0, 255), 2)

        # Guardar la imagen con las minutias en la carpeta "minutias"
        minutiae_image_path = os.path.join('minutias', os.path.basename(image_path).replace(".tif",".jpg"))
        cv2.imwrite(minutiae_image_path, image_with_minutiae)

    return image_with_contours, contours, minutiae, contours_image_path, minutiae_image_path

# Ruta de la carpeta con las imágenes
dataset_folder = 'dataset'

# Listas para almacenar los datos
filenames = []
contours_image_paths = []
minutiae_image_paths = []
etiquetas = []

# Extraer las minutias y mostrar los contornos y minutias de cada imagen en la carpeta
for filename in os.listdir(dataset_folder):
    if filename.endswith('.tif'):
        image_path = os.path.join(dataset_folder, filename)
        image_contours, contours, minutiae, contours_image_path, minutiae_image_path = extract_minutiae(image_path, save_images=True)

        # Determinar la etiqueta de la imagen basándose en el nombre del archivo
        etiqueta = int(filename.split('_')[0]) - 101

        # Agregar los datos a las listas
        filenames.append(filename)
        contours_image_paths.append(contours_image_path)
        minutiae_image_paths.append(minutiae_image_path)
        etiquetas.append(etiqueta)

        # Mostrar el número de minutias extraídas y las coordenadas de las minutias
        print("Minutias extraídas de", filename)
        print("Número de minutias extraídas:", len(minutiae))
        for i, (start, end) in enumerate(minutiae):
            print("Minutia", i+1, ": Start:", start, "End:", end)

        print("-------------------------------------")

# Crear el archivo CSV con la información de las imágenes y etiquetas
csv_path = 'datos.csv'
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image', 'Contour Image', 'Minutiae Image', 'Label'])
    for i in range(len(filenames)):
        writer.writerow([filenames[i], contours_image_paths[i], minutiae_image_paths[i], etiquetas[i]])

print("Archivo CSV creado con éxito:", csv_path)
