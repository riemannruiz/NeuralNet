import numpy as np

from libraries_.Grad_Desc_training_ import Grad_Desc_training_
from libraries_.f_sigmoid import f_sigmoid
from libraries_.neural_model_ import neural_model
from libraries_.f_lineal import f_lineal
from libraries_.ini_network import ini_network
from libraries_.cost_fun_rmse_reg_ import cost_fun_rmse_reg_

# Generar los datos para el aprendizaje de la red neuronal


# Generar datos de entrada y salida
x1 = np.arange(-1, 1, 0.01)
y = 4 * x1 * np.sin(10 * x1) + x1 * np.exp(x1) * np.cos(20 * x1)

# Organizar datos en forma de tabla
X = x1.reshape(-1, 1)
Y = y.reshape(-1, 1)

# Inicializar parámetros de la red neuronal
params = ini_network(X, Y, [10, 5])
params.hidden_fun[0] = f_sigmoid

# Modificar tasa de aprendizaje
params.eta = 0.1
# Verificar las dimensiones de X
print("Dimensiones de X:", X.shape)

# Verificar las dimensiones de W
# Verificar las dimensiones de cada matriz en params.W
for k, W_k in enumerate(params.W):
    print(f"Dimensiones de params.W[{k}]:", np.array(W_k).shape)

# Verificar las dimensiones de b
# Verificar las dimensiones de cada matriz en params.b
for k, b_k in enumerate(params.b):
    print(f"Dimensiones de params.b[{k}]:", np.array(b_k).shape)

# Obtener estimación de la red neuronal
Yhat,_ = neural_model(X, params)

# Calcular la función de costo RMSE
J = cost_fun_rmse_reg_(Y, Yhat, params)

# Entrenamiento de la red neuronal
n_iter = 5000
params, model = Grad_Desc_training_(X, Y, params, n_iter)

# Los resultados finales se encuentran en model.inputs[-1]
Y2 = model.inputs[-1]

#Traceback (most recent call last):

 #   params, model = Grad_Desc_training_(X, Y, params, n_iter)
                    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 # File "C:\Users\Ricardo\Desktop\nn2\libraries_\Grad_Desc_training_.py", line 44, in Grad_Desc_training_
  #  inputs, hidden_out, phi_ = neural_layer(model.inputs[-1], params.W[k], params.b[k], params.hidden_fun[k])
                          #                                                              ~~~~~~~~~~~~~~~~~^^^
#IndexError: list index out of range
