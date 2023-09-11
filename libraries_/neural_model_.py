from libraries_.ini_network import Parameters
from libraries_.neural_layer_ import neural_layer
import numpy as np
class Model:
    def __init__(self):
        self.inputs = []      # Lista para almacenar los datos de entrada en cada capa.
        self.hidden_out = []  # Lista para almacenar las salidas de las capas ocultas.


def neural_model(X, params):
    n_capas = len(params.W)  # Numero de capas de la red neuronal (se incluye la capa de salida)
    # Creacion de instancia Model
    model = Model()
    model.inputs.append(X) # almacenar temporalmente la informacion propagada

    for k in range(n_capas - 1):
        model.inputs.append(None) # Agregar un valor None a la lista de entradas para la capa actual
        model.hidden_out.append(None) # Agregar un valor None a la lista de salidas ocultas para la capa actual
        inputs_k, _, hidden_out_k = neural_layer(  # Ajusta la asignación para desempaquetar los tres valores
            model.inputs[k], params.W[k], params.b[k], params.hidden_fun[k])
        model.inputs[k + 1] = inputs_k # Almacenar las entradas calculadas en la lista de entradas para la próxima capa
        model.hidden_out[k] = hidden_out_k # Capas ocultas

    k = n_capas - 1
    model.inputs.append(None)
    model.hidden_out.append(None)
    model.inputs[k + 1], _, _ = neural_layer(  # Ajusta la asignación para desempaquetar los tres valores
        model.inputs[k], params.W[k], params.b[k], params.out_fun)  # Capas de salida

    Y = np.array(model.inputs[k + 1]) # Obtener las salidas finales como un array numpy
    model_ = model
    return Y, model_