import numpy as np
from libraries_.f_lineal import f_lineal
from libraries_.f_tanh import f_tanh
from libraries_.cost_fun_rmse_reg_ import cost_fun_rmse_reg_

class Parameters:
    def __init__(self):
        self.n_in = None
        self.n_samples = None
        self.n_hidden = None
        self.n_out = None
        self.output_delay = 0
        self.nw_total = None
        self.nw_total_layer = None
        self.nw_total_bias = None
        self.batch_size = None
        self.W = []
        self.b = []
        self.hidden_fun = []
        self.Wr = []
        self.out_fun = None
        self.loss = None
        self.lambda_reg = None
        self.eta = None
        self.mu = [0.5] 
        self.mu_f = None
        self.mu_limits = None
        self.batch_size = None
        self.save_dW = None
        self.tol_loss = None
        self.tol_dJ = None
        self.view_process = None
        self.mu_update_limit = None
        self.indx_W_Jac = {}



def ini_network(X = None, Y =None, neurons_hidden_layer = None, **kwargs):
    params = Parameters()
    if X is None or Y is None or neurons_hidden_layer is None:
        raise ValueError(
            'Se requiere definir como mínimo las variables siguientes: X = datos de entrada, Y = datos de salida, neurons_hidden_layer = estructura interna de la red')

    params.output_delay = 0

    # Asignar los parametros definidos por el usuario
    if 'output_delay' in kwargs:
        params.output_delay = kwargs['output_delay']

    if 'invalid_parameter' in kwargs:
        print(f"Imposible asignar {kwargs['invalid_parameter']}. Se usarán valores predeterminados")

    params.n_in = X.shape[1] # numero de entradas a la red neuronal
    params.n_samples = X.shape[0]  # numero de muestras en total
    params.n_hidden = neurons_hidden_layer # numero de salidas de la red neuronal
    params.n_out = Y.shape[1]
    size_W = [params.n_in] + neurons_hidden_layer + [params.n_out]
    nw_total = 0
    nw_total_layer = 0
    nw_total_bias = 0

    for k in range(len(size_W) - 2):
        params.b.append(np.random.rand(size_W[k + 1], 1) - 0.5)  # Inicializacion del bias de la capa
        params.W.append(np.random.rand(size_W[k + 1], size_W[k]) - 0.5)  # Inicializacion de pesos de cada capa
        params.hidden_fun.append(f_tanh)  # funcion de activacion de la capa oculta
        nw_total += (size_W[k] + 1) * size_W[k + 1]  #  cantidad total de pesos en las capas ocultas
        nw_total_layer += size_W[k] * size_W[k + 1]  #cantidad total de pesos en capas ocultas
        nw_total_bias += size_W[k + 1]  # cantidad total de bias en capas ocultas

        if k == 0 and params.output_delay > 0:
            params.Wr.append(np.random.rand(size_W[k + 1], params.output_delay * params.n_out) - 0.5)
            nw_total += (params.output_delay * params.n_out) * size_W[k + 1]

    k += 1
    params.nw_total = nw_total + (size_W[k] + 1) * size_W[k + 1]  # cantidad total de pesos en la red neuronal
    params.nw_total_layer = nw_total_layer + size_W[k] * size_W[k + 1]  # cantidad total de bias en capas ocultas
    params.nw_total_bias = nw_total_bias + size_W[k + 1]  # cantidad total de bias en capas ocultas
    params.b.append(np.random.rand(size_W[k + 1], 1) - 0.5)  # Inicializacion del bias de la capa de salida
    params.W.append(np.random.rand(size_W[k + 1], size_W[k]) - 0.5)  #  Inicializacion de pesos de la capa de salida
    params.out_fun = f_lineal  # funcion de activacion de la capa de salida
    params.loss = cost_fun_rmse_reg_
    params.lambda_reg = 0  # Inicializacion del coeficiente de regularizacion
    params.eta = 0.1  # Inicializacion del coeficiente de aprendizaje
    params.mu_f = 2  # Factor de rescalamiento para el entrenamiento Levemberg-Marquart
    params.mu_limits = [1e-5, 10000]  # Limites para el valor de mu en el entrenamiento Levemberg-Marquart
    params.batch_size = 10000  # Cantidad de datos usada para cada actualizacion
    params.save_dW = False  # Indicacion para guardar los deltas del entrenamiento
    params.tol_loss = 1e-4  # Toleranacia para el valor de la funcion de costo
    params.tol_dJ = 1e-4  # Tolerancia para el valor del gradiente de la funcion de costo
    params.view_process = True  # Bandera de activacion para mostrar en cosola el avance del entrenamiento
    params.mu_update_limit = 50  # Limite de actualizaciones para el valor de mu

    #  Creacion de la estructura con los indices de los pesos dentro del Jacobiano
    idx0_w = 1
    for k in range(len(params.W)-1,-1,-1):
        if k == 0 and params.output_delay > 0:
            f, c = params.W[k].shape
        else:
            f, c = params.W[k].shape

        idx1_w = idx0_w - 1 + f * (c + 1)
        params.indx_W_Jac[k] = list(range(idx0_w-1, idx1_w))
        idx0_w = idx1_w + 1

    return params