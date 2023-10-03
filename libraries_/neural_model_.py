from libraries_.ini_network import Parameters
from libraries_.model import Model
from libraries_.neural_layer_ import neural_layer


def neural_model(X, params):
    n_capas = len(params.W)  # Numero de capas de la red neuronal (se incluye la capa de salida)
    # Creacion de instancia Model
    model = Model()
    model.inputs = [None for _ in range(n_capas + 1)]
    model.hidden_out = [None for _ in range(n_capas)]

    # Propagación de la información por la red neuronal
    model.inputs[0] = X  # Inicializa la entrada
    # Capas ocultas
    for k in range(n_capas - 1):
        model.inputs[k + 1], model.hidden_out[k], _ = neural_layer(model.inputs[k], params.W[k], params.b[k],
                                                                   params.hidden_fun[k])

        # Imprime los tamaños para verificar

    # Capa de salida
    k = n_capas - 1
    model.inputs[k + 1], model.hidden_out[k], _ = neural_layer(model.inputs[k], params.W[k], params.b[k],
                                                               params.out_fun)
    Y = model.inputs[k + 1]
    model_ = model
    return Y, model_
