import pandas as pd
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import keyboard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Limpiar la consola
os.system('cls' if os.name == 'nt' else 'clear')

# Rutas de los archivos CSV
train_file = 'train_data.csv'
val_file = 'val_data.csv'
test_file = 'test_data.csv'

# Leer los archivos CSV en DataFrames
train_data = pd.read_csv(train_file)
val_data = pd.read_csv(val_file)
test_data = pd.read_csv(test_file)

# Tamaño deseado de las imágenes
target_size = (50, 50)

# Función para cargar y redimensionar las imágenes
def load_and_preprocess_image(image_path):
    image = load_img(image_path, target_size=target_size)  # Redimensionar la imagen al tamaño deseado
    image_array = img_to_array(image) / 255.0  # Normalizar los valores de píxel entre 0 y 1
    return image_array

# Obtener las rutas de las imágenes de entrenamiento, las etiquetas correspondientes y convertir las etiquetas en one-hot encoding
train_images = [load_and_preprocess_image(image_path) for image_path in train_data['Ruta']]
train_images = np.array(train_images)
train_labels = tf.keras.utils.to_categorical(train_data['Etiqueta'])

# Obtener las rutas de las imágenes de validación, las etiquetas correspondientes y convertir las etiquetas en one-hot encoding
val_images = [load_and_preprocess_image(image_path) for image_path in val_data['Ruta']]
val_images = np.array(val_images)
val_labels = tf.keras.utils.to_categorical(val_data['Etiqueta'])

# Obtener las rutas de las imágenes de prueba y las etiquetas correspondientes
test_images = [load_and_preprocess_image(image_path) for image_path in test_data['Ruta']]
test_images = np.array(test_images)
test_labels = tf.keras.utils.to_categorical(test_data['Etiqueta'])

# Crear el modelo de la red neuronal
model = Sequential()
model.add(Flatten(input_shape=train_images.shape[1:]))  # Aplanar la imagen en un vector
#model.add(Dropout(.5))
model.add(Dense(256, activation='relu'))
#model.add(Dropout(.5))
model.add(Dense(256, activation='relu'))
#model.add(Dropout(.5))
model.add(Dense(128, activation='relu'))
#model.add(Dropout(.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(40, activation='softmax'))  # Capa de salida con activación softmax



# Definir una función para detener el entrenamiento
def stop_training():
    model.stop_training = True

# Asignar la función stop_training a la tecla F10
keyboard.on_press_key("f10", lambda _: stop_training())

# Entrenar el modelo
desired_acc = 1

# Definir una función para detener el entrenamiento cuando se alcanza la precisión objetivo en entrenamiento y validación
class StopTrainingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') >= desired_acc and logs.get('val_accuracy') >= desired_acc:
            self.model.stop_training = True


# Compilar el modelo
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo con la callback StopTrainingCallback
history = model.fit(train_images, train_labels, epochs=1000, batch_size=1024, validation_data=(val_images, val_labels),
                    callbacks=[StopTrainingCallback()])


# Guardar el modelo

# Cargar los datos y las etiquetas verdaderas
test_data = pd.read_csv('test_data.csv')
test_images = np.array([load_and_preprocess_image(image_path) for image_path in test_data['Ruta']])
test_labels = tf.keras.utils.to_categorical(test_data['Etiqueta'])

# Obtener las predicciones y la pérdida del modelo
loss, accuracy = model.evaluate(test_images, test_labels)

# Obtener las predicciones del modelo
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Calcular las métricas
accuracy = accuracy_score(np.argmax(test_labels, axis=1), predicted_labels)
precision = precision_score(np.argmax(test_labels, axis=1), predicted_labels, average='macro')
recall = recall_score(np.argmax(test_labels, axis=1), predicted_labels, average='macro')
f1 = f1_score(np.argmax(test_labels, axis=1), predicted_labels, average='macro')
confusion = confusion_matrix(np.argmax(test_labels, axis=1), predicted_labels)
classification = classification_report(np.argmax(test_labels, axis=1), predicted_labels)
loss, accuracy = model.evaluate(test_images, test_labels)

# Imprimir las métricas
print('Test loss:', loss)
print('Test accuracy:', accuracy)
print("Loss:", loss)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("Confusion Matrix:\n", confusion)
print("Classification Report:\n", classification)

model.save('modelo.h5')

# Mostrar el gráfico del historial de precisión
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


