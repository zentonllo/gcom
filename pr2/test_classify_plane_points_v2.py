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
import time

from mlp import MLP

__author__ = "Ignacio Casso, Daniel Gamo, Gwydion J. Martín, Alberto Terceño"

# create data

nb_black = 100
nb_red = 100
nb_data = nb_black + nb_red

s = np.linspace(0, 4 * np.pi, nb_black)

x_black = np.vstack([np.log(1 + s) * np.cos(s),
                     np.log(1 + s) * np.sin(s)]).T
x_red = np.vstack([-np.log(1 + s) * np.cos(s),
                   -np.log(1 + s) * np.sin(s)]).T

x_data = np.vstack((x_black, x_red))

t_data = np.asarray([0] * nb_black + [1] * nb_red).reshape(nb_data, 1)


#  Net structure

D = x_data.shape[1]  # initial dimension
K = 1  # final dimension

#  You must find the best MLP structure in order to
#  misclassify as few points as possible. Training time will be
#  measured too.
#  You can use at most 3000 weights and 2000 epochs.
#  For example:

K_list = [D, 10, 10, 10, 10, 10, 10, K]  # list of dimensions of layers

activation_functions = [np.tanh,
                        np.tanh,
                        np.tanh,
                        np.tanh,
                        np.tanh,
                        np.tanh,
                        np.tanh]

diff_activation_functions = [MLP.dtanh,
                             MLP.dtanh,
                             MLP.dtanh,
                             MLP.dtanh,
                             MLP.dtanh,
                             MLP.dtanh,
                             MLP.dtanh]

# network training

# List of different learning (epsilons) and regularization (betas) rates
# used in this benchmark (to test some values just change the lower and upper
# values and the number of values evenly spaced between these bounds

lower_epsilon = 0.1
upper_epsilon = 0.1
n_epsilons = 1

lower_beta = 0.001
upper_beta = 0.001
n_betas = 1


learning_rates = np.linspace(lower_epsilon, upper_epsilon, n_epsilons)
regularization_rates = np.linspace(lower_beta, upper_beta, n_betas)

# We will only accept results under this threshold
# (that is, total number of misclassified points)
min_wrongs = 200

# Iterate over the different rates
for epsilon_test in learning_rates:
    for beta_test in regularization_rates:
        time_begin = time.time()

        mlp = MLP(K_list,
                  activation_functions, diff_activation_functions)

        mlp.train(x_data, t_data,
                  epochs=2000, batch_size=10,
                  epsilon=epsilon_test,
                  beta=beta_test,
                  print_cost=False)

        time_end = time.time()

        # check if circles nearby the middle one are
        # correctly classified

        mlp.get_activations_and_units(x_black)
        wrong_black = (mlp.y > 1 / 2).squeeze()

        mlp.get_activations_and_units(x_red)
        wrong_red = (mlp.y < 1 / 2).squeeze()

        miss_red = np.sum(wrong_red)
        miss_black = np.sum(wrong_black)
        missclassified = miss_red + miss_black

        #  Prepare plots and save current best parameters
        if (missclassified < min_wrongs):
            min_wrongs = missclassified
            min_wrong_black = miss_red
            min_wrong_red = miss_black
            best_epsilon = epsilon_test
            best_beta = beta_test
            time_best_train = time_end - time_begin
            # plot the probability mapping and the data

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

            plt.scatter(x_black[wrong_black, 0],
                        x_black[wrong_black, 1],
                        facecolors='None',
                        s=50,
                        marker='o', color='red')

            plt.scatter(x_red[wrong_red, 0],
                        x_red[wrong_red, 1],
                        facecolors='None',
                        s=50,
                        marker='o', color='black')


# Print the optimal parameters
print('Time used in training %f' % time_best_train)
print('Optimal epsilon parameter: %f' % best_epsilon)
print('Optimal beta (regularization) parameter: %f' % best_beta)
print('Points misclassified as red: {}'.format(min_wrong_red))
print('Points misclassified as black: {}'.format(min_wrong_black))
plt.show()
