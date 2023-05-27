def seguir_contorno(imagen_binaria, x_inicial, y_inicial):
    contorno = []
    x_actual = x_inicial
    y_actual = y_inicial

    while True:
        contorno.append((x_actual, y_actual))
        imagen_binaria[y_actual, x_actual] = 255  #Marca el píxel como visitado (blanco)

        vecinos = obtener_vecinos(imagen_binaria, x_actual, y_actual)
        vecinos_activos = [(x, y) for x, y in vecinos if imagen_binaria[y, x] == 0]

        if vecinos_activos:
            x_actual, y_actual = vecinos_activos[0]
        else:
            break

    return contorno

def obtener_vecinos(imagen_binaria, x, y):
    vecinos = []
    altura, anchura = imagen_binaria.shape

    if x > 0:
        vecinos.append((x - 1, y))
    if x < anchura - 1:
        vecinos.append((x + 1, y))
    if y > 0:
        vecinos.append((x, y - 1))
    if y < altura - 1:
        vecinos.append((x, y + 1))

    return vecinos

def encontrar_contornos(imagen_binaria):
    contornos = []
    altura, anchura = imagen_binaria.shape

    for y in range(altura):
        for x in range(anchura):
            if imagen_binaria[y, x] == 0:  #Píxel negro (contorno)
                contorno = seguir_contorno(imagen_binaria, x, y)
                contornos.append(contorno)

    return contornos