#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Test MLP class for regression

@author: avaldes
"""
from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
import time

from mlp import MLP


__author__ = "Ignacio Casso, Daniel Gamo, Gwydion J. Martín, Alberto Terceño"
#  create data


def f1(x):
    return 1 / (1 + x**2)


def f2(x):
    return np.sin(x)


nb_data = 100
x_data = np.linspace(-5, 5, nb_data).reshape(nb_data, 1)
t_data1 = f1(x_data)
t_data2 = f2(x_data)

#  Net structure

D = 1  # initial dimension
K = 1  # final dimension

#  You must find the best MLP structure in order to
#  obtain the least possible L2 error. Training time will be
#  measured too.
#  You can use at most 1000 weights and 1000 epochs.
#  For example:

K_list = [D, 20, 20, 10, 10, K]  # list of dimensions of layers

activation_functions = [MLP.sigmoid,
                        MLP.sigmoid,
                        MLP.sigmoid,
                        MLP.sigmoid,
                        MLP.identity]

diff_activation_functions = [MLP.dsigmoid,
                             MLP.dsigmoid,
                             MLP.dsigmoid,
                             MLP.dsigmoid,
                             MLP.didentity]


# network training

for t_data in [t_data1, t_data2]:

    time_begin = time.time()

    mlp = MLP(K_list,
              activation_functions, diff_activation_functions)

    mlp.train(x_data, t_data,
              epochs=1000, batch_size=10,
              epsilon=0.1,
              print_cost=True)

    time_end = time.time()

    print('Time used in training: {}'.format(time_end - time_begin))

    mlp.get_activations_and_units(x_data)
    print('L2 ERROR={}'.format(np.sum(t_data - mlp.y)**2))
    plt.plot(x_data, mlp.y)
    plt.plot(x_data, t_data, color='black')
    plt.show()
