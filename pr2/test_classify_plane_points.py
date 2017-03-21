#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Checking the implementation of MLP
Creates two concentric circles and checks whether
points near the middle circle are correctly classified.

@author: avaldes
"""
from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
import time

from mlp import MLP

# create data

nb_black = 50
nb_red = 50
nb_data = nb_black + nb_red

s = np.linspace(0, 2 * np.pi, nb_black)

x_black = np.vstack([np.cos(s), np.sin(s)]).T +\
    np.random.randn(nb_black, 2) * 0
x_red = np.vstack([2 * np.cos(s), 2 * np.sin(s)]).T +\
    np.random.randn(nb_red, 2) * 0

x_middle = (x_black + x_red) / 2

x_data = np.vstack((x_black, x_red))

t_data = np.asarray([0] * nb_black + [1] * nb_red).reshape(nb_data, 1)


#  Net structure

D = x_data.shape[1]  # initial dimension
K = 1  # final dimension

#  You must find the best MLP structure in order to
#  misclassify as few points as possible. Training time will be
#  measured too.
#  You can use at most 1000 weights and 1000 epochs.
#  For example:

K_list = [D, 10, 10, 10, 5, K]  # list of dimensions of layers

activation_functions = [MLP.sigmoid,
                        MLP.sigmoid,
                        MLP.sigmoid,
                        MLP.sigmoid,
                        MLP.sigmoid]

diff_activation_functions = [MLP.dsigmoid,
                             MLP.dsigmoid,
                             MLP.dsigmoid,
                             MLP.dsigmoid,
                             MLP.dsigmoid]

# network training

time_begin = time.time()

mlp = MLP(K_list,
          activation_functions, diff_activation_functions)

mlp.train(x_data, t_data,
          epochs=1000, batch_size=10,
          epsilon=0.1,
          beta=0.001,
          print_cost=True)

time_end = time.time()

print('Time used in training %f' % (time_end - time_begin))


# check if circles nearby the middle one are
# correctly classified

dr = 0.1
r1 = 3 / 2 - dr
r2 = 3 / 2 + dr

s_test = np.linspace(0, 2 * np.pi, 100)
black_pts_test = r1 * np.array([np.cos(s_test), np.sin(s_test)]).T
red_pts_test = r2 * np.array([np.cos(s_test), np.sin(s_test)]).T

mlp.get_activations_and_units(black_pts_test)
wrong_black = (mlp.y > 1 / 2).squeeze()
print('Points misclassified as black:%f' % np.sum(wrong_black))

mlp.get_activations_and_units(red_pts_test)
wrong_red = (mlp.y < 1 / 2).squeeze()
print('Points misclassified as red:%f' % np.sum(wrong_red))


# plot the probability mapping and
# the data

delta = 0.01
x = np.arange(-3, 3, delta)
y = np.arange(-3, 3, delta)


X, Y = np.meshgrid(x, y)
x_pts = np.vstack((X.flatten(), Y.flatten())).T
mlp.get_activations_and_units(x_pts)
grid_size = X.shape[0]
Z = mlp.y.reshape(grid_size, grid_size)

plt.axis('equal')
plt.contourf(X, Y, Z, 50)
plt.scatter(x_black[:, 0], x_black[:, 1],
            marker=',',
            s=1,
            color='black')
plt.scatter(x_red[:, 0], x_red[:, 1],
            marker=',',
            s=1,
            color='red')

plt.scatter(x_middle[:, 0], x_middle[:, 1], marker='.', color='white')
plt.scatter(black_pts_test[:, 0], black_pts_test[:, 1],
            marker='+', color='black')
plt.scatter(black_pts_test[wrong_black, 0],
            black_pts_test[wrong_black, 1],
            marker='+', color='red')
plt.scatter(red_pts_test[:, 0], red_pts_test[:, 1],
            marker='+', color='red')
plt.scatter(red_pts_test[wrong_red, 0],
            red_pts_test[wrong_red, 1],
            marker='+', color='black')


plt.show()
