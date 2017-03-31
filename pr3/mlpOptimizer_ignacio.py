# -*- coding: utf-8 -*-
"""

@author: Alberto
"""

from __future__ import division, print_function

import numpy as np


__author__ = "Ignacio Casso, Daniel Gamo, Gwydion J. Martín, Alberto Terceño"


class Optimizer(object):

    def process_batch(self, x_data, t_data):
        print("metodo no implementado")

    dic_learning_methods = {} #No se puede inicializar aqui porque todavia no estan definidos

    @staticmethod
    def get_optimizer(mlp, **kwargs):

        method_name = kwargs.pop("method", "SGD")
        method = Optimizer.dic_learning_methods[method_name]
        return method(mlp, **kwargs)

class SGD(Optimizer):

    def __init__(self, mlp, **kwargs):
        self.mlp = mlp

        self.epsilon = kwargs.get("epsilon", 0.01)
        self.beta = kwargs.get("beta", 0)

    def process_batch(self, x_data, t_data):

        grad_w_list, grad_b_list = self.mlp.get_gradients(x_data, t_data, self.beta)

        self.mlp.weights_list = [w - self.epsilon*grad_w
                                 for w, grad_w in zip(self.mlp.weights_list, grad_w_list)]
        self.mlp.biases_list = [b - self.epsilon*grad_b
                                for b, grad_b in zip(self.mlp.biases_list, grad_b_list)]

Optimizer.dic_learning_methods["SGD"] = SGD


class Momentum(Optimizer):
    
    def __init__(self, mlp, **kwargs):
        self.mlp = mlp

        self.epsilon = kwargs.get("epsilon", 0.01)
        self.gamma = kwargs.get("gamma", 0.9)
        self.beta = kwargs.get("beta", 0)

        self.v_w_list = [0]*mlp.nb_layers
        self.v_b_list = [0]*mlp.nb_layers

    def process_batch(self, x_data, t_data):
    
        grad_w_list, grad_b_list = self.mlp.get_gradients(x_data, t_data, self.beta)

        self.v_w_list = [self.gamma*v_w + self.epsilon*grad_w for v_w, grad_w in zip(self.v_w_list, grad_w_list)]
        self.v_b_list = [self.gamma*v_b + self.epsilon*grad_b for v_b, grad_b in zip(self.v_b_list, grad_b_list)]

        self.mlp.weights_list = [w - v_w for w, v_w in zip(self.mlp.weights_list, self.v_w_list)]
        self.mlp.biases_list = [b - v_b for b, v_b in zip(self.mlp.biases_list, self.v_b_list)]

Optimizer.dic_learning_methods["Momentum"] = Momentum



class Nesterov(Optimizer):

    def __init__(self, mlp, **kwargs):
        self.mlp = mlp

        self.epsilon = kwargs.get("epsilon", 0.01)
        self.gamma = kwargs.get("gamma", 0.9)
        self.beta = kwargs.get("beta", 0)

        self.v_w_list, self.v_b_list = [0]*mlp.nb_layers, [0]*mlp.nb_layers
    
    def process_batch(self, x_data,t_data):

        w_aux_list, b_aux_list = self.mlp.weights_list, self.mlp.biases_list

        self.mlp.weights_list = [w - self.gamma*v_w
                                 for w, v_w in zip(self.mlp.weights_list, self.v_w_list)]

        self.mlp.biases_list = [b - self.gamma*v_b
                                for b, v_b in zip(self.mlp.biases_list, self.v_b_list)]            

        grad_w_list, grad_b_list = self.mlp.get_gradients(x_data, t_data, self.beta)

        self.v_w_list = [self.gamma * v_w + self.epsilon*grad_w for v_w, grad_w in zip (self.v_w_list, grad_w_list)]
        self.v_b_list = [self.gamma * v_b + self.epsilon*grad_b for v_b, grad_b in zip(self.v_b_list, grad_b_list)]

        self.mlp.weights_list = [w_aux - self.epsilon*v_w for w_aux, v_w in zip(w_aux_list, self.v_w_list)]
        self.mlp.biases_list = [b_aux - self.epsilon*v_b for b_aux, v_b in zip(b_aux_list, self.v_b_list)]

Optimizer.dic_learning_methods["Nesterov"] = Nesterov



class Adagrad(Optimizer):

    def __init__(self, mlp, **kwargs):
        self.mlp = mlp

        self.epsilon = kwargs.get("epsilon", 0.01)        
        self.beta = kwargs.get("beta", 0)
        self.ep = kwargs.get("ep", 0.00000001)

        self.G_w_list = [0]*mlp.nb_layers
        self.G_b_list = [0]*mlp.nb_layers

    def process_batch(self, x_data, t_data):

        grad_w_list, grad_b_list = self.mlp.get_gradients(x_data, t_data, self.beta)

        self.G_w_list = [G_w + grad_w*grad_w for G_w, grad_w in zip(self.G_w_list, grad_w_list)]
        self.G_b_list = [G_b+ grad_b*grad_b for G_b, grad_b in zip(self.G_b_list, grad_b_list)]

        self.mlp.weigths_list = [w - (self.epsilon/np.sqrt(G_w+self.ep))*grad_w for w, G_w, grad_w in
                                zip(self.mlp.weights_list, self.G_w_list, grad_w_list)]

        self.mlp.biases_list = [b - (self.epsilon/np.sqrt(G_b+self.ep))*grad_b for b, G_b, grad_b in
                                zip(self.mlp.biases_list, self.G_b_list, grad_b_list)]

Optimizer.dic_learning_methods["Adagrad"] = Adagrad

