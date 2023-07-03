import os
import shutil

ruta_actual = os.path.dirname(os.path.abspath(__file__))

# Ruta del directorio padre
directorio_padre = os.path.abspath(os.path.join(ruta_actual, ".."))

# Ruta de las carpetas de origen
carpeta1 = os.path.join(directorio_padre, "DB1_B")
carpeta2 = os.path.join(directorio_padre, "DB2_B")
carpeta3 = os.path.join(directorio_padre, "DB3_B")
carpeta4 = os.path.join(directorio_padre, "DB4_B")

# Ruta de la carpeta de destino
carpeta_destino = os.path.join(directorio_padre, "dataset")

# Eliminar el contenido de la carpeta de destino
if os.path.exists(carpeta_destino):
    shutil.rmtree(carpeta_destino)
os.makedirs(carpeta_destino)

# Función para obtener un nuevo nombre si hay conflicto
def obtener_nuevo_nombre(nombre_archivo, carpeta):
    nuevo_nombre = nombre_archivo
    contador = 1
    while os.path.exists(os.path.join(carpeta_destino, nuevo_nombre)):
        # Extraer el número y el label del nombre de archivo existente
        partes = nuevo_nombre.split("_")
        numero = int(partes[0])
        label = partes[1].split(".")[0]
        
        # Generar un nuevo nombre incrementando el número
        numero += 1
        nuevo_nombre = f"{numero}_{label}.tif"
        
        # Verificar si ya existe un archivo con ese nombre en la carpeta
        if not os.path.exists(os.path.join(carpeta_destino, nuevo_nombre)):
            break
        contador += 1
    
    return nuevo_nombre

# Combina los archivos de las carpetas de origen en la carpeta de destino
for archivo in os.listdir(carpeta1):
    ruta_archivo = os.path.join(carpeta1, archivo)
    nuevo_nombre = obtener_nuevo_nombre(archivo, carpeta_destino)
    shutil.copy(ruta_archivo, os.path.join(carpeta_destino, nuevo_nombre))

for archivo in os.listdir(carpeta2):
    ruta_archivo = os.path.join(carpeta2, archivo)
    nuevo_nombre = obtener_nuevo_nombre(archivo, carpeta_destino)
    shutil.copy(ruta_archivo, os.path.join(carpeta_destino, nuevo_nombre))

for archivo in os.listdir(carpeta3):
    ruta_archivo = os.path.join(carpeta3, archivo)
    nuevo_nombre = obtener_nuevo_nombre(archivo, carpeta_destino)
    shutil.copy(ruta_archivo, os.path.join(carpeta_destino, nuevo_nombre))

for archivo in os.listdir(carpeta4):
    ruta_archivo = os.path.join(carpeta4, archivo)
    nuevo_nombre = obtener_nuevo_nombre(archivo, carpeta_destino)
    shutil.copy(ruta_archivo, os.path.join(carpeta_destino, nuevo_nombre))

print("La combinación de las carpetas se ha completado con éxito.")
