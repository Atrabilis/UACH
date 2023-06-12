import matplotlib.pyplot as plt

def grafo_linea_desicion(w, b, inputs, labels):
    # Cálculo de las coordenadas x1 correspondientes a los límites del gráfico
    x0_min = -1
    x1_min = (-(w[0] * x0_min) - b[0]) / w[1]

    x0_max = 2
    x1_max = (-(w[0] * x0_max) - b[0]) / w[1]

    # Graficar la línea de decisión y los puntos de datos
    plt.plot([x0_min, x0_max], [x1_min, x1_max])  # Graficar la línea de decisión
    plt.scatter(inputs[labels == 0, 0], inputs[labels == 0, 1], label='class 0', marker='o')  # Graficar los puntos de clase 0
    plt.scatter(inputs[labels == 1, 0], inputs[labels == 1, 1], label='class 1', marker='s')  # Graficar los puntos de clase 1

    # Configuración de la leyenda y visualización del gráfico
    plt.legend(loc='best')  # Mostrar la leyenda en la esquina superior derecha
    plt.show()  # Mostrar el gráfico
