import numpy as np
from libraries_.model import Model
from libraries_.neural_layer_ import neural_layer
from libraries_.neural_model_ import neural_model
from libraries_.delta_hidden_ import delta_hidden
from libraries_.delta_output import delta_output

import numpy as np


def Grad_Desc_training_(X, Y, params, max_iter):
    n_samples = X.shape[0]  # número de muestras en el conjunto de datos
    lambda_ = params.lambda_reg  # coeficiente de regularización
    eta = params.eta  # coeficiente de aprendizaje
    bias = np.ones((n_samples, 1))
    n_capas = len(params.W)  # Número de capas de la red neuronal (sin incluir la capa de salida)
    model = Model()  # Usaremos la clase Model para almacenar los resultados
    
    # Esta seccion es esclusiva de la implementacion de python
    model.inputs = ['']*(n_capas+1) # Inicializacion de los espacios para guardar los inputs
    model.hidden_out = ['']*n_capas
    model.phi_ = ['']*n_capas
    model.loss = ['']*max_iter
    model.deltas = ['']*n_capas
    model.b_gradient = ['']*n_capas
    model.W_gradient = ['']*n_capas
    model.d_b = ['']*n_capas
    model.d_W = ['']*n_capas
    
    
    model.inputs[0] = X  # estructura para almacenar temporalmente la información propagada
    params.W0 = params.W.copy()  # Guardar las condiciones iniciales de los pesos
    params.b0 = params.b.copy()  # Guardar las condiciones iniciales del bias

    n_disp = round(max_iter * 0.05)  # mostrará el avance cada 5% de avance

    iter_ = 1
    while iter_ <= max_iter:
        # Simulación de la red neuronal (información hacia adelante)
        # Capas ocultas

        for k in range(n_capas-1):
            # Antes de la llamada a neural_layer
            # print(f"Dimensiones en capa {k + 1}:")
            # print("X:", model.inputs[-1].shape)
            # print(f"params.W[{k}]:", params.W[k].shape)
            # print(f"params.b[{k}]:", params.b[k].shape)

            model.inputs[k+1], model.hidden_out[k], model.phi_[k] = neural_layer(model.inputs[k], params.W[k], params.b[k], params.hidden_fun[k])            

            
        k = n_capas-1
        model.inputs[k+1], model.hidden_out[k], model.phi_[k] = neural_layer(model.inputs[k], params.W[k], params.b[k], params.out_fun)

        

        # Almacenar la función de costo en cada iteración
        model.loss[iter_-1], dJ_dY,_,_ = params.loss(Y, model.inputs[-1], params)


        # Propagación de la información hacia atrás para calcular deltas
        k = n_capas-1
        model.deltas[k] = delta_output(Y,model.inputs[k+1],model.phi_[k],params)
        model.b_gradient[k] = model.deltas[k].T@bias
        
        

        # # Cálculo de Delta y el Gradiente de las capas ocultas
        # for k in range(n_capas - 1, -1, -1):
        #     model.deltas.append(delta_hidden(model.phi_[k], params.W[k + 1], model.deltas[-1]))
        #     model.b_gradient.append(model.deltas[-1].T @ bias)
        #     model.W_gradient.append(model.deltas[-1].T @ model.inputs[k] + lambda_ * params.W[k])
        #     model.d_b.append(-eta * model.b_gradient[-1])
        #     model.d_W.append(-eta * model.W_gradient[-1])

        # # Actualización de los pesos
        # for k_update in range(n_capas):
        #     params.b[k_update] += model.d_b.pop()  # Actualización del bias
        #     params.W[k_update] += model.d_W.pop()  # Actualización de los pesos

        # # Guardar el historial de los cambios de los pesos
        # if params.save_dW:
        #     model.dW_hist.append(model.d_W.copy())
        #     model.db_hist.append(model.d_b.copy())

        # if iter_ % n_disp == 0 and params.view_process:
        #     print(
        #         f'Progress {100 * (iter_ / max_iter):.1f}%: {iter_} epocas was reached,  with a loss {model.loss[-1]:.3f}')

        iter_ += 1

        # if model.loss[-1] <= params.tol_loss or np.mean(np.abs(dJ_dY)) <= params.tol_dJ:
        #     print('The loss function or gradient reach the desired values.')
        #     break

    # if params.view_process:
    #     print(f'{iter_ - 1} epocas was reached, with a loss {model.loss[-1]:.3f}')

    return params, model