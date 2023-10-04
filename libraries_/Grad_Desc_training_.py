import numpy as np
from libraries_.model import Model
from libraries_.neural_layer_ import neural_layer
from libraries_.delta_output import delta_output
from libraries_.delta_hidden_ import delta_hidden_


def Grad_Desc_training_(X, Y, params, max_iter):
    n_samples = X.shape[0]  # número de muestras en el conjunto de datos
    lambda_ = params.lambda_reg  # coeficiente de regularización
    eta = params.eta  # coeficiente de aprendizaje
    bias = np.ones((n_samples, 1))
    n_capas = len(params.W)  # Número de capas de la red neuronal (sin incluir la capa de salida)
    model = Model()  # Usaremos la clase Model para almacenar los resultados

    model.inputs = [np.array([]) for _ in range(n_capas + 1)]
    model.hidden_out = [np.array([]) for _ in range(n_capas)]
    model.phi_ = [np.array([]) for _ in range(n_capas)]
    model.loss = np.ones(max_iter) * 10000  # Inicializar como una lista de números de punto flotante en lugar de cadenas
    model.deltas = [np.array([]) for _ in range(n_capas)]
    model.b_gradient = [np.array([]) for _ in range(n_capas)]
    model.W_gradient = [np.array([]) for _ in range(n_capas)]
    model.d_b = [np.zeros((params.W[k].shape[0], 1)) for k in range(n_capas)]
    model.d_W = [np.zeros_like(params.W[k]) for k in range(n_capas)]

    model.inputs[0] = X  # estructura para almacenar temporalmente la información propagada
    params.W0 = params.W.copy()  # Guardar las condiciones iniciales de los pesos
    params.b0 = params.b.copy()  # Guardar las condiciones iniciales del bias

    n_disp = max(1, round(max_iter * 0.05)) # Se asegura que n_disp nunca sea igual a cero

    iter_ = 0  # Cambio el valor inicial de iter_
    while iter_ < max_iter:  # Cambio la condición del bucle
        # Simulación de la red neuronal (información hacia adelante)
        # Capas ocultas
        for k in range(n_capas - 1):
            model.inputs[k + 1], model.hidden_out[k], model.phi_[k] = neural_layer(model.inputs[k], params.W[k],
                                                                                   params.b[k], params.hidden_fun[k])

        k = n_capas - 1
        model.inputs[k + 1], model.hidden_out[k], model.phi_[k] = neural_layer(model.inputs[k], params.W[k],
                                                                               params.b[k], params.out_fun)

        model.loss[iter_], dJ_dY, _, _ = params.loss(Y, model.inputs[-1], params)  # Cambio el índice aquí

        # Propagación de la información hacia atrás para calcular deltas
        k = n_capas - 1
        model.deltas[k] = delta_output(Y, model.inputs[k + 1], model.phi_[k], params)
        model.b_gradient[k] = model.deltas[k].T @ bias
        model.W_gradient[k] = model.deltas[k].T @ model.inputs[k] + lambda_ * params.W[k]
        model.d_b[k] = -eta * model.b_gradient[k]
        model.d_W[k] = -eta * model.W_gradient[k]

        for k in range(n_capas - 2, -1, -1):
            model.deltas[k] = delta_hidden_(model.phi_[k], params.W[k + 1], model.deltas[k + 1])
            model.b_gradient[k] = model.deltas[k].T @ bias
            model.W_gradient[k] = model.deltas[k].T @ model.inputs[k] + lambda_ * params.W[k]
            model.d_b[k] = -eta * model.b_gradient[k]
            model.d_W[k] = -eta * model.W_gradient[k]

        # Actualizacion de los pesos
        for k in range(n_capas):
            params.b[k] = params.b[k] + model.d_b[k]
            params.W[k] = params.W[k] + model.d_W[k]

        if params.save_dW:
            model.dW_hist[iter_] = model.d_W
            model.db_hist[iter_] = model.d_b

        if iter_ % n_disp == 0 and params.view_process:
            progress = 100 * (iter_ / max_iter)
            print(f'Progress {progress:.1f}%: {iter_} epocs was reached, with a loss {model.loss[iter_]:.3f}')

        if iter_ == max_iter and params.view_process:
            print(f'Progress 100%: {max_iter} epocs was reached, with a loss {model.loss[iter_]:.3f}')  

        iter_ += 1

    if params.view_process:
        print(f'{iter_} epocs was reached, with a loss {model.loss[iter_-1]:.3f}')
    
    
    return params, model