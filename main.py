import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from libraries_.Grad_Desc_training_ import Grad_Desc_training_
from libraries_.f_sigmoid import f_sigmoid
from libraries_.neural_model_ import neural_model
from libraries_.ini_network import ini_network
from libraries_.Levenberg_marquart_training import Levenberg_Marquart_training_

from libraries_.cost_fun_rmse_reg_ import cost_fun_rmse_reg_
import time


#%% Generar datos de entrada y salida
x1 = np.arange(-1, 1.01, 0.01)
y = 4 * x1 * np.sin(10 * x1) + x1 * np.exp(x1) * np.cos(20 * x1)

# Organizar datos en forma de tabla
X = x1.reshape(-1, 1)
Y = y.reshape(-1, 1)

# #%% Cargar los archivos CSV en DataFrames
# df_W1 = pd.read_csv('W1.csv', header=None)  
# df_W2 = pd.read_csv('W2.csv', header=None)
# df_W3 = pd.read_csv('W3.csv', header=None)
# df_b1 = pd.read_csv('b1.csv', header=None)
# df_b2 = pd.read_csv('b2.csv', header=None)
# df_b3 = pd.read_csv('b3.csv', header=None)

# # Convierte el DataFrame en una matriz NumPy
# W1 = df_W1.to_numpy()  
# W2 = df_W2.to_numpy()
# W3 = df_W3.to_numpy()
# b1 = df_b1.to_numpy()
# b2 = df_b2.to_numpy()
# b3 = df_b3.to_numpy()




params = ini_network(X, Y, [10, 5])

#params.hidden_fun[0] = f_sigmoid  
params.eta = 0.1

# params.W[0] = W1
# params.W[1] = W2
# params.W[2] = W3
# params.b[0] = b1
# params.b[1] = b2
# params.b[2] = b3
#%%
# Obtener estimación de la red neuronal
Yhat, model = neural_model(X, params)


# Calcular la función de costo RMSE
J, _, _, _ = cost_fun_rmse_reg_(Y, Yhat, params)

#%% Entrenamiento de la red neuronal

n_iter = 50
start_time = time.time()
# params, model = Grad_Desc_training_(X, Y, params, n_iter)
params, model = Levenberg_Marquart_training_(X, Y, params, n_iter)

end_time = time.time()

# Calcular el tiempo transcurrido en segundos
elapsed_time = end_time - start_time

# Imprime el tiempo transcurrido
print(f"Elapsed time is: {elapsed_time:.6f} seconds")

# Los resultados finales se encuentran en model.inputs[-1]
Y2 = model.inputs[-1]
J = model.loss

# %% Dibujar resultados
plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
plt.plot(X,Y,c='b')
plt.plot(X,Y2,c='r')
plt.xlabel('x_1'),plt.ylabel('y'),plt.title('Y_real vs Y_estimated')
plt.subplot(1,2,2)
plt.scatter(np.arange(len(J)),J,c='b')
plt.xlabel('epochs'),plt.ylabel('J(.)'),plt.title('Cost function')
plt.show()