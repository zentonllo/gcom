# -*- coding: utf-8 -*-
"""

@author: Alberto
"""

from __future__ import division, print_function

import sys
import numpy as np
import mlp

__author__ = "Ignacio Casso, Daniel Gamo, Gwydion J. Martín, Alberto Terceño"

class Optimizer(object):
    
    def __init__(self, mpercep, method, epsilon, beta, gamma):
        
        self.epsilon = epsilon
        self.beta = beta
        self.gamma = gamma
        self.mpercep = mpercep
        
        self.method = method
        
        # Fatla switch para inicializar los argumentos adecuados
        self.sgd_args = {}
        #  self.momentum_args = {"v_w":[0]*self.mpercep.nb_layers, "v_b":[0]*self.mpercep.nb_layers}
        # self.nesterov_args = {"v_w":[0]*self.mpercep.nb_layers, "v_b":[0]*self.mpercep.nb_layers}
        self.v_w = [0]*self.mpercep.nb_layers
        self.v_b = [0]*self.mpercep.nb_layers
      
    
    def run(self,x_data,t_data,indexes):
        if self.method == 'SGD':
            self.sgd(x_data,t_data,indexes)
        elif self.method == 'Momentum':
            self.momentum(x_data,t_data, indexes, self.v_w, self.v_b)
        else :
            self.nesterov(x_data, t_data, indexes, self.v_w, self.v_b)
    
    def sgd(self,x_data,t_data,indexes): 
        self.mpercep.get_gradients(x_data[indexes], t_data[indexes], self.beta)
        self.mpercep.weights_list = [(self.mpercep.weights_list[k] -
                                 self.epsilon*self.mpercep.grad_w_list[k])
                                 for k in range(self.mpercep.nb_layers)]
        self.mpercep.biases_list = [(self.mpercep.biases_list[k] -
                                self.epsilon*self.mpercep.grad_b_list[k])
                                for k in range(self.mpercep.nb_layers)]
    
    
    def momentum(self,x_data,t_data, indexes, v_w, v_b):

        self.mpercep.get_gradients(x_data[indexes], t_data[indexes], self.beta)
        for k in range(self.mpercep.nb_layers):
            v_w[k] = self.gamma * v_w[k] + self.epsilon*self.mpercep.grad_w_list[k]
            v_b[k] = self.gamma * v_b[k] + self.epsilon*self.mpercep.grad_b_list[k]

            self.mpercep.weights_list[k] = (self.mpercep.weights_list[k] -
                                	        self.epsilon*self.mpercep.grad_w_list[k])
            self.mpercep.biases_list[k] = (self.mpercep.biases_list[k] -
                            	               self.epsilon*self.mpercep.grad_b_list[k])
        ### Cambio a v_w = [self.gamma * v_w[k] + self.epsilon*self.mpercep.grad_w_list[k] for k in range(self.mpercep.nb_layers)]??
    
    def nesterov(self, x_data,t_data,indexes, v_w, v_b):

        w_aux, b_aux = self.mpercep.weights_list, self.mpercep.biases_list

        self.mpercep.weights_list = [self.mpercep.weights_list[k] - self.gamma*v_w[k]
                                 for k in range(self.mpercep.nb_layers)]
        self.mpercep.biases_list = [self.mpercep.biases_list[k] - self.gamma*v_b[k]
                                for k in range(self.mpercep.nb_layers)]            

        self.mpercep.get_gradients(x_data[indexes], t_data[indexes], self.beta)

        for k in range(self.mpercep.nb_layers):

            v_w[k] = self.gamma * v_w[k] + self.epsilon*self.mpercep.grad_w_list[k]
            v_b[k] = self.gamma * v_b[k] + self.epsilon*self.mpercep.grad_b_list[k]

            self.mpercep.weights_list[k] = w_aux[k] - self.epsilon*v_w[k]
            self.mpercep.biases_list[k] = b_aux[k] - self.epsilon*v_b[k]
            
            # Quitar el for como antes?

