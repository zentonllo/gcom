#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Checking the implementation of MLP
Data in points along two logarithmic spirals
@author: avaldes
"""

from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
# import time

from mlp import MLP

# create data

nb_black = 100
nb_red = 100
nb_data = nb_black + nb_red

s = np.linspace(0, 4*np.pi, nb_black)

x_black = np.vstack([np.log(1 + s) * np.cos(s),
                     np.log(1 + s) * np.sin(s)]).T
x_red = np.vstack([-np.log(1 + s) * np.cos(s),
                   -np.log(1 + s) * np.sin(s)]).T

x_data = np.vstack((x_black, x_red))

t_data = np.asarray([0]*nb_black + [1]*nb_red).reshape(nb_data, 1)


#  Net structure

D = x_data.shape[1]  # initial dimension
K = 1  # final dimension

#  You must find the best MLP structure in order to
#  misclassify as few points as possible. Training time will be
#  measured too.
#  You can use at most 3000 weights and 2000 epochs.
#  For example:

K_list = [D, 20, 20, 20, K]  # list of dimensions of layers

activation_functions = [MLP.relu] * 3 + [MLP.sigmoid]

diff_activation_functions = [MLP.drelu] * 3

# network training
methods = ['SGD', 'momentum', 'nesterov', 'adagrad',
           'adadelta', 'RMS_prop', 'adam']

fig, ax = plt.subplots(2, 4)

list_pairs = [(r, c) for r in range(2) for c in range(4)]

for counter, method in enumerate(methods):
    method = methods[counter]
    print(method)
    mlp = MLP(K_list,
              activation_functions,
              diff_activation_functions,
              init_seed=5)

    mlp.train(x_data, t_data,
              epochs=2000, batch_size=20,
              eta=0.01,
              beta=0,
              method=method,
              print_cost=True,
              initialize_weights=True)

    delta = 0.05
    a, b = -4, 4
    x = np.arange(a, b, delta)
    y = np.arange(a, b, delta)

    X, Y = np.meshgrid(x, y)
    x_pts = np.vstack((X.flatten(), Y.flatten())).T
    mlp.get_activations_and_units(x_pts)

    grid_size = X.shape[0]
    Z = mlp.y.reshape(grid_size, grid_size)
    
    r, c = list_pairs[counter]
    curr_axes = ax[r, c]
    curr_axes.axis('equal')
    curr_axes.contourf(X, Y, Z, 50)

    curr_axes.scatter(x_data[:, 0], x_data[:, 1],
                      marker='o',
                      s=1,
                      color='black')
    curr_axes.set_xlim(-4, 4)
    curr_axes.set_ylim(-4, 4)
    curr_axes.set_xlabel(method)


plt.show()
