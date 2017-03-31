# -*- coding: utf-8 -*-
"""

@author: Alberto
"""

from __future__ import division, print_function


__author__ = "Ignacio Casso, Daniel Gamo, Gwydion J. Martín, Alberto Terceño"


class Optimizer(object):

    def __init__(self, mpercep, method, epsilon, beta, gamma):

        self.epsilon = epsilon
        self.beta = beta
        self.gamma = gamma
        self.mpercep = mpercep

        self.method = method

        # Lista de estructura de datos adicionales
        self.v_w = None
        self.v_b = None

        self.init_extra_data_structures()
    
    
    def init_extra_data_structures(self):
        #  Initialize Momentum and Nesterov args
        if self.method == 'Momentum' or self.method == 'Nesterov':
            self.v_w = [0] * self.mpercep.nb_layers
            self.v_b = [0] * self.mpercep.nb_layers
    
    def run(self, x_data, t_data):
        if self.method == 'SGD':
            self.sgd(x_data, t_data)
        elif self.method == 'Momentum':
            self.momentum(x_data, t_data)
        else:
            self.nesterov(x_data, t_data)

    def sgd(self, x_data, t_data):
        self.mpercep.get_gradients(x_data, t_data, self.beta)
        self.mpercep.weights_list = [(self.mpercep.weights_list[k] -
                                      self.epsilon * self.mpercep.grad_w_list[k])
                                     for k in range(self.mpercep.nb_layers)]
        self.mpercep.biases_list = [(self.mpercep.biases_list[k] -
                                     self.epsilon * self.mpercep.grad_b_list[k])
                                    for k in range(self.mpercep.nb_layers)]

    def momentum(self, x_data, t_data):

        self.mpercep.get_gradients(x_data, t_data, self.beta)
        self.v_w = [self.gamma * self.v_w[k] +
                    self.epsilon * self.mpercep.grad_w_list[k] for k in range(self.mpercep.nb_layers)]
        self.v_b = [self.gamma * self.v_b[k] +
                    self.epsilon * self.mpercep.grad_b_list[k] for k in range(self.mpercep.nb_layers)]
        self.mpercep.weights_list = [(
            self.mpercep.weights_list[k] - self.v_w[k]) for k in range(self.mpercep.nb_layers)]
        self.mpercep.biases_list = [(
            self.mpercep.biases_list[k] - self.v_b[k]) for k in range(self.mpercep.nb_layers)]

    def nesterov(self, x_data, t_data):

        w_aux, b_aux = self.mpercep.weights_list, self.mpercep.biases_list

        self.mpercep.weights_list = [self.mpercep.weights_list[k] - self.gamma * self.v_w[k]
                                     for k in range(self.mpercep.nb_layers)]
        self.mpercep.biases_list = [self.mpercep.biases_list[k] - self.gamma * self.v_b[k]
                                    for k in range(self.mpercep.nb_layers)]

        self.mpercep.get_gradients(x_data, t_data, self.beta)

        self.v_w = [self.gamma * self.v_w[k] +
                    self.epsilon * self.mpercep.grad_w_list[k] for k in range(self.mpercep.nb_layers)]
        self.v_b = [self.gamma * self.v_b[k] +
                    self.epsilon * self.mpercep.grad_b_list[k] for k in range(self.mpercep.nb_layers)]
        self.mpercep.weights_list = [
            w_aux[k] - self.epsilon * self.v_w[k] for k in range(self.mpercep.nb_layers)]
        self.mpercep.biases_list = [
            b_aux[k] - self.epsilon * self.v_b[k] for k in range(self.mpercep.nb_layers)]
