# -*- coding: utf-8 -*-
"""
Test MLP class for regression

@author: avaldes
"""
from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt


from mlp import MLP

def f1(x):
    return np.cos(5*x)

nb_data = 500
x_data = np.linspace(-5, 5, nb_data).reshape(nb_data, 1)
t_data = f1(x_data)

D = 1
K = 1

K_list = [D, 500, K]  # list of dimensions of layers

activation_functions = [MLP.sigmoid] * 1 + [MLP.identity]
diff_activation_functions = [MLP.dsigmoid] * 1

plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_aspect('equal')
nb_epochs = 1000

mlp = MLP(K_list,
          activation_functions,
          diff_activation_functions,
          init_seed=6)

for epoch in range(nb_epochs):
    mlp.train(x_data, t_data,
              epochs=1, batch_size=100,
              eta=0.1,
              method='adam',
              gamma=0.9,
              beta=0,
              beta_1=0.99,
              beta_2=0.999,
              initialize_weights=(epoch == 0),
              print_cost=True)

    mlp.get_activations_and_units(x_data)

    error = mlp.cost_L2(mlp.y, t_data)
    if epoch == 0:
        true_dib = ax.plot(x_data, t_data)
        dib, = ax.plot(x_data, mlp.y)
    else:
        dib.set_ydata(mlp.y)
        fig.canvas.draw()
        if 0:
            plt.savefig('picture'+str(epoch))

