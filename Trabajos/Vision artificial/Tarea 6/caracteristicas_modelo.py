def caracteristicas_modelo(perceptron,inputs, labels, nombre_del_modelo):
    presicion = perceptron.evaluate(inputs, labels)
    print("Presicion {} = {}%".format(nombre_del_modelo, presicion*100))
    print("Pesos de {} = {}".format(nombre_del_modelo, perceptron.weights))