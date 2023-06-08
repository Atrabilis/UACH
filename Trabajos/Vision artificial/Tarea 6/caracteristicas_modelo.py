def caracteristicas_modelo(perceptron, inputs, labels, nombre_del_modelo):
    # Evaluación del modelo
    precision = perceptron.evaluate(inputs, labels)
    
    # Impresión de la precisión y los pesos del modelo
    print("\nPrecisión {} = {}%\n".format(nombre_del_modelo, precision*100))
    print("Pesos de {} = {}".format(nombre_del_modelo, perceptron.weights))
