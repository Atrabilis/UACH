import tensorflow as tf
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split

# Ruta de la carpeta con las imágenes en formato TIFF
tiff_folder = 'dataset'

# Ruta de la carpeta donde se guardarán las imágenes convertidas
output_folder = 'ruta_de_la_carpeta_de_salida'

# Crear la carpeta de salida si no existe
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Recorrer las imágenes en formato TIFF en la carpeta
for filename in os.listdir(tiff_folder):
    if filename.endswith('.tif'):
        tiff_path = os.path.join(tiff_folder, filename)
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.png')  # Cambiar la extensión a PNG

        # Abrir la imagen TIFF y guardarla en formato PNG
        image = Image.open(tiff_path)
        image.save(output_path, 'PNG')

# Obtener la lista de imágenes convertidas
image_paths = [os.path.join(output_folder, filename) for filename in os.listdir(output_folder) if filename.endswith('.png')]

# Obtener las etiquetas de las imágenes basadas en los nombres de archivo
labels = [int(filename.split('_')[0])-101 for filename in os.listdir(output_folder) if filename.endswith('.png')]

# Dividir los datos en conjuntos de entrenamiento, validación y prueba
train_data, test_data, train_labels, test_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42)
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

# Función para cargar una imagen y preprocesarla
def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (128, 128))
    return image

# Función para obtener el número de clases
def get_num_classes(labels):
    return len(set(labels))

# Cargar las imágenes de entrenamiento, validación y prueba
train_images = [load_image(image_path) for image_path in train_data]
val_images = [load_image(image_path) for image_path in val_data]
test_images = [load_image(image_path) for image_path in test_data]

# Convertir las etiquetas a tensores
train_labels = tf.convert_to_tensor(train_labels)
val_labels = tf.convert_to_tensor(val_labels)
test_labels = tf.convert_to_tensor(test_labels)

# Crear el conjunto de datos de entrenamiento, validación y prueba
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

# Configurar el tamaño del lote y mezclar los datos de entrenamiento
batch_size = 32
train_dataset = train_dataset.shuffle(len(train_images)).batch(batch_size)
val_dataset = val_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

# Crear el modelo de red neuronal
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(128, 128, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(get_num_classes(labels), activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Entrenar el modelo
epochs = 10
model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)

# Evaluar el modelo en el conjunto de prueba
test_loss, test_accuracy = model.evaluate(test_dataset)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)
