# -*- coding: utf-8 -*-
"""

@author: Alberto
"""

from __future__ import division, print_function

import numpy as np


__author__ = "Ignacio Casso, Daniel Gamo, Gwydion J. Martín, Alberto Terceño"


class Optimizer(object):
    
    
    @staticmethod
    def get_optimizer(mlp, **kwargs):

        dic_learning_methods = {'SGD': SGD, 'Momentum': Momentum,
                                'Nesterov': Nesterov, 'Adagrad': Adagrad}

        method_name = kwargs.pop("method", "SGD")
        method = dic_learning_methods[method_name]
        return method(mlp, **kwargs)


class SGD(Optimizer):

    def __init__(self, mlp, **kwargs):
        self.mlp = mlp

        self.epsilon = kwargs.pop("epsilon", 0.01)
        self.beta = kwargs.pop("beta", 0)

    def process_batch(self, x_data, t_data):

        grad_w_list, grad_b_list = self.mlp.get_gradients(
            x_data, t_data, self.beta)

        self.mlp.weights_list = [w - self.epsilon * grad_w
                                 for w, grad_w in zip(self.mlp.weights_list, grad_w_list)]
        self.mlp.biases_list = [b - self.epsilon * grad_b
                                for b, grad_b in zip(self.mlp.biases_list, grad_b_list)]


class Momentum(Optimizer):

    
    def init_aux_structures(self,mlp):
        self.v_w_list = []
        self.v_b_list = []

        for layer in range(mlp.nb_layers):
            new_v_w = np.zeros((mlp.K_list[layer], mlp.K_list[layer + 1]))
            new_v_b = np.zeros(mlp.K_list[layer + 1])
            self.v_w_list.append(new_v_w)
            self.v_b_list.append(new_v_b)

        
    
    def __init__(self, mlp, **kwargs):
        self.mlp = mlp

        self.epsilon = kwargs.pop("epsilon", 0.01)
        self.gamma = kwargs.pop("gamma", 0.9)
        self.beta = kwargs.pop("beta", 0)
        
        self.init_aux_structures(mlp)

    def process_batch(self, x_data, t_data):

        grad_w_list, grad_b_list = self.mlp.get_gradients(
            x_data, t_data, self.beta)

        self.v_w_list = [self.gamma * v_w + self.epsilon *
                         grad_w for v_w, grad_w in zip(self.v_w_list, grad_w_list)]
        self.v_b_list = [self.gamma * v_b + self.epsilon *
                         grad_b for v_b, grad_b in zip(self.v_b_list, grad_b_list)]

        self.mlp.weights_list = [
            w - v_w for w, v_w in zip(self.mlp.weights_list, self.v_w_list)]
        self.mlp.biases_list = [b - v_b for b,
                                v_b in zip(self.mlp.biases_list, self.v_b_list)]


class Nesterov(Optimizer):
    
    def init_aux_structures(self,mlp):
        self.v_w_list = []
        self.v_b_list = []

        for layer in range(mlp.nb_layers):
            new_v_w = np.zeros((mlp.K_list[layer], mlp.K_list[layer + 1]))
            new_v_b = np.zeros(mlp.K_list[layer + 1])
            self.v_w_list.append(new_v_w)
            self.v_b_list.append(new_v_b)
    
    
    
    def __init__(self, mlp, **kwargs):
        self.mlp = mlp

        self.epsilon = kwargs.pop("epsilon", 0.01)
        self.gamma = kwargs.pop("gamma", 0.9)
        self.beta = kwargs.pop("beta", 0)
        
        self.init_aux_structures(mlp) 

    def process_batch(self, x_data, t_data):

        w_aux_list, b_aux_list = self.mlp.weights_list, self.mlp.biases_list

        self.mlp.weights_list = [w - self.gamma * v_w
                                 for w, v_w in zip(self.mlp.weights_list, self.v_w_list)]

        self.mlp.biases_list = [b - self.gamma * v_b
                                for b, v_b in zip(self.mlp.biases_list, self.v_b_list)]

        grad_w_list, grad_b_list = self.mlp.get_gradients(
            x_data, t_data, self.beta)

        self.v_w_list = [self.gamma * v_w + self.epsilon *
                         grad_w for v_w, grad_w in zip(self.v_w_list, grad_w_list)]
        self.v_b_list = [self.gamma * v_b + self.epsilon *
                         grad_b for v_b, grad_b in zip(self.v_b_list, grad_b_list)]

        self.mlp.weights_list = [
            w_aux - self.epsilon * v_w for w_aux, v_w in zip(w_aux_list, self.v_w_list)]
        self.mlp.biases_list = [b_aux - self.epsilon *
                                v_b for b_aux, v_b in zip(b_aux_list, self.v_b_list)]


class Adagrad(Optimizer):

    def init_aux_structures(self,mlp):
        self.G_w_list = []
        self.G_b_list = []

        for layer in range(mlp.nb_layers):
            new_G_w = np.zeros((mlp.K_list[layer], mlp.K_list[layer + 1]))
            new_G_b = np.zeros(mlp.K_list[layer + 1])
            self.G_w_list.append(new_G_w)
            self.G_b_list.append(new_G_b)
    
    
    def __init__(self, mlp, **kwargs):
        self.mlp = mlp

        self.epsilon = kwargs.pop("epsilon", 0.01)
        self.beta = kwargs.pop("beta", 0)
        self.ep = kwargs.pop("ep", 1e-8)
        
        self.init_aux_structures(mlp) 
        # self.G_w_list = [0] * mlp.nb_layers
        # self.G_b_list = [0] * mlp.nb_layers

    def process_batch(self, x_data, t_data):

        grad_w_list, grad_b_list = self.mlp.get_gradients(
            x_data, t_data, self.beta)

        self.G_w_list = [G_w + (grad_w ** 2) for G_w,
                         grad_w in zip(self.G_w_list, grad_w_list)]
        self.G_b_list = [G_b + (grad_b ** 2) for G_b,
                         grad_b in zip(self.G_b_list, grad_b_list)]

        self.mlp.weigths_list = [w - (self.epsilon / np.sqrt(G_w + self.ep)) * grad_w for w, G_w, grad_w in
                                 zip(self.mlp.weights_list, self.G_w_list, grad_w_list)]

        self.mlp.biases_list = [b - (self.epsilon / np.sqrt(G_b + self.ep)) * grad_b for b, G_b, grad_b in
                                zip(self.mlp.biases_list, self.G_b_list, grad_b_list)]
