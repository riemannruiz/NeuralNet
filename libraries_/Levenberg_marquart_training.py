import numpy as np
from libraries_.model import Model
from libraries_.neural_layer_ import neural_layer
from libraries_.neural_model_ import  neural_model
from libraries_.jacobian_k_opt_v2_ import Jacobian_k_opt_v2

def Levenberg_Marquart_training_(X, Y, params, max_iter):
    model = Model()
    n_samples = X.shape[0]  # número de muestras en el conjunto de datos
    lambda_ = params.lambda_reg
    mu = params.mu[0]  # parametro Levemberg-Marquart
    beta = params.mu_f  # factor de reescalamiento de mu
    n_capas = len(params.W)  # Número de capas de la red neuronal
    model.inputs = [np.array([]) for _ in range(n_capas + 1)]

    model.hidden_out = [np.array([]) for _ in range(n_capas)]
    model.phi_ = [np.array([]) for _ in range(n_capas)]
    model.loss = np.ones(max_iter) * 10000  # Inicializar como una lista de números de punto flotante en lugar de cadenas
    model.inputs[0] = X
    params.W0 = params.W.copy()
    params.b0 = params.b.copy()

    n_disp = max(1, round(max_iter * 0.01))

    print('%%%%%%%%%%%%%%%%%%%%%%%')
    print('Training Initiated')
    print('%%%%%%%%%%%%%%%%%%%%%%%')
    print(f"Training {params.nw_total} weights, with {max_iter} epochs")
    print('.......................')

    iter_ = 1  # contador inicial de epocas
    Jloss0 = np.inf
    while iter_ <= max_iter:
        # simulacion de la red neuronal ( informacion hacia adelante)
        for k in range(n_capas - 1):
            model.inputs[k + 1], model.hidden_out[k], model.phi_[k] = neural_layer(model.inputs[k], params.W[k],
                                                                                   params.b[k], params.hidden_fun[k])
        # Capa de salida
        k = n_capas - 1
        model.inputs[k + 1], model.hidden_out[k], model.phi_[k] = neural_layer(model.inputs[k], params.W[k],
                                                                                   params.b[k], params.out_fun)
        model.loss[iter_], _, _, E_Jac = params.loss(Y, model.inputs[-1], params)

        # Propagacion de la informacion hacia atras por medio del calculo de deltas
        # Inicializacion de la matriz Jacobiana llena de ceros
        Jac = np.zeros((params.nw_total, params.n_out * n_samples))
        #  Respaldo de la estructura minima de la red original, porque mas
        # adelante se hara el recorrido de cada salida por separado para el
        # calculo del Jacobiano.

        params_ = params
        inputs_end = model.inputs[-1]
        hidden_out = model.hidden_out[-1]  # Salidas ocultas de la ultima capa
        phi_out = model.phi_[-1]  # Derivadas de la ultima capa
        for s in range(Y.shape[1]):
            model.inputs[-1] = inputs_end[:, s]
            model.hidden_out[-1] = hidden_out[:, s]
            model.phi_[-1] = phi_out[:, s]
            params.W[-1] = params_.W[-1][s, :].reshape(1, -1)
            params.b[-1] = params_.b[-1][s, :].reshape(1, -1)

            Jac[:, (s - 1) * n_samples:s * n_samples] = Jacobian_k_opt_v2(Y[:, s], model, params, s)

        # Restauracion de los parametros originales de la red neuronal (la modificacion se hizo solo para el calcula del Jacobiano)
        params = params_
        model.inputs[-1] = inputs_end
        model.hidden_out[-1] = hidden_out
        model.phi_[-1] = phi_out

        # Respaldo de los valores importantes de la actualizacion

        J_loss_ = model.loss[iter_]
        W_ = params.W
        b_ = params.b
        k_mu_updates = 0  # Variable para contar el numero de veces que se ha actulizado. Sirve para detener el algoritmo

        while J_loss_ >= model.loss[iter_]:
            k_mu_updates = k_mu_updates + 1
            # Obtener los pesos en vector columna para agregarlo como termino
            # de regularizacion
            Wtot = np.zeros((params.nw_total, 1))

            for layer in range(n_capas, 0, -1):
                Wtmp = np.concatenate((np.zeros((params.W[layer - 1].shape[0], 1)), params.W[layer - 1]), axis=1)
                Wtot[params.indx_W_Jac[layer - 1], 0] = Wtmp.flatten()
            dW = np.linalg.inv(Jac.T @ Jac + mu * np.eye(params.nw_total)) @ (Jac.T @ E_Jac.flatten() + lambda_ * Wtot)

            # Actualizacion donde se considera la adaptacion de \mu.
            for k in range(n_capas, 0, -1):
                f, c = params.W[k].shape  # Dimension de los pesos de la capa

                d_temp = np.reshape(dW[params.indx_W_Jac[k]], (f, c + 1)).T
                model.d_b[k] = d_temp[:, 0]  # Guardar el delta_b de la capa
                model.d_W[k] = d_temp[:, 1:]  # Guardar el delta_W de la capa
                params.b[k] += model.d_b[k]  # Actualizacion de bias
                params.W[k] += model.d_W[k]  # Actualizacion de pesos

            # Calculo de la nueva funcion de costo


            Y_hat_ = neural_model(model.inputs, params)
            J_loss_ = params.loss(Y, Y_hat_, params)
            # Actualizacion del parametro mu

            if J_loss_ >= model.loss[iter_]:
                mu = mu * beta
                params.W = W_  # Recuperar los pesos antes de la actualizacion
                params.b = b_  # Recuperar los pesos antes de la actualizacion
                if mu > params.mu_limits[1]:
                    mu = params.mu_limits[1]
                else:
                    mu = mu / beta
                    if mu < params.mu_limits[0]:
                        mu = params.mu_limits[0]

                if k_mu_updates > params.mu_update_limit:
                    print(f'The algorithm reached the limit for the mu updates: {k_mu_updates} updates')
                    break

        # Guardar el historial de los cambios de los pesos

        if params.save_dW:
            model.dW_hist[iter_] = model.d_W
        # Guardar el comportamiento de los paramnetros
        params.mu[iter_] = mu
        # Avances en las iteraciones del algoritmo

        if iter_ % n_disp == 0 and params.view_process:
            progress = 100 * (iter / max_iter)
            print(f'Progress {progress:.1f}%: {iter} epochs were reached, with a loss {model.loss[iter_]:.6f}')

        # Criterio de paro del algoritmo de entrenamiento
        if model.loss[iter_] <= params.tol_loss or abs(model.loss[iter_] - Jloss0) <= params.tol_dJ:
            print('The loss function or gradient reached the desired values.')
            print(f'Optimized values: (Loss: {model.loss[iter_]:.4f}, Gradient: {abs(model.loss[iter_] - Jloss0):.4f})  Limited: (Loss: {params.tol_loss}, Gradient: {params.tol_dJ})')
            break

        Jloss0 = model.loss[iter_]

        iter = iter_ + 1

    if params.view_process:
        print(f'{iter_} epochs were reached, with a loss {model.loss[iter_ - 1]:.3f}')

    return params, model
