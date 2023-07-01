import os
import shutil

ruta_actual = os.path.dirname(os.path.abspath(__file__))
directorio_padre = os.path.abspath(os.path.join(ruta_actual, ".."))
carpeta_destino = os.path.join(directorio_padre, "dataset")

# Eliminar todo el contenido de la carpeta destino antes de comenzar
shutil.rmtree(carpeta_destino, ignore_errors=True)
os.makedirs(carpeta_destino)

# Ruta de las carpetas de origen
carpeta1 = os.path.join(directorio_padre, "DB1_B")
carpeta2 = os.path.join(directorio_padre, "DB2_B")
carpeta3 = os.path.join(directorio_padre, "DB3_B")
carpeta4 = os.path.join(directorio_padre, "DB4_B")

# Combina los archivos de las carpetas de origen en la carpeta de destino
def combinar_carpeta(carpeta_origen):
    for archivo in os.listdir(carpeta_origen):
        ruta_archivo_origen = os.path.join(carpeta_origen, archivo)
        ruta_archivo_destino = os.path.join(carpeta_destino, archivo)
        shutil.copy(ruta_archivo_origen, ruta_archivo_destino)

# Combinar carpetas
combinar_carpeta(carpeta1)
combinar_carpeta(carpeta2)
combinar_carpeta(carpeta3)
combinar_carpeta(carpeta4)

print("La combinación de las carpetas se ha completado con éxito.")
