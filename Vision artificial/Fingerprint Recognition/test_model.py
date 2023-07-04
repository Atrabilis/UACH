import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Cargar el modelo guardado
model = load_model('modelo_entrenado.h5')

# Función para cargar y preprocesar la imagen
def load_and_preprocess_image(image_path):
    image = image = load_img(image_path, target_size=(50, 50), color_mode='grayscale')  # Asegurarse de que la imagen tenga el mismo tamaño que las imágenes de entrenamiento
    image_array = img_to_array(image) / 255.0  # Normalizar los valores de píxel entre 0 y 1
    return image_array

# Directorio que contiene las imágenes
dataset_dir = 'contornos'

# Obtener la lista de archivos en el directorio
image_files = os.listdir(dataset_dir)

# Iterar sobre cada imagen y realizar la predicción
for image_file in image_files:
    # Construir la ruta completa de la imagen
    image_path = os.path.join(dataset_dir, image_file)
    
    # Cargar y preprocesar la imagen
    image = load_and_preprocess_image(image_path)
    
    # Realizar la predicción
    predictions = model.predict(np.expand_dims(image, axis=0))
    
    # Obtener la clase predicha (índice con mayor probabilidad)
    predicted_class = np.argmax(predictions)
    
    # Obtener la probabilidad asociada a la clase predicha
    confidence = predictions[0][predicted_class]
    
    # Imprimir los resultados
    print('Imagen:', image_file)
    print('Clase predicha:', predicted_class)
    print('Probabilidad:', confidence)
    print('---')
