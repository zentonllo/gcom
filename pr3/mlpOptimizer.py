# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np


__author__ = "Ignacio Casso, Daniel Gamo, Gwydion J. Martín, Alberto Terceño"


class Optimizer(object):
    
    
    @staticmethod
    def get_optimizer(mlp, **kwargs):

        dic_learning_methods = {'SGD': SGD, 'momentum': Momentum,
                                'nesterov': Nesterov, 'adagrad': Adagrad,
                                'adadelta':Adadelta, 'RMS_prop': RMSprop, 
                                'adam':Adam }

        method_name = kwargs.pop("method", "SGD")
        method = dic_learning_methods[method_name]
        return method(mlp, **kwargs)


class SGD(Optimizer):

    def __init__(self, mlp, **kwargs):
        self.mlp = mlp

        self.eta = kwargs.pop("eta", 0.1)
        self.beta = kwargs.pop("beta", 0)

    def process_batch(self, x_data, t_data):

        grad_w_list, grad_b_list = self.mlp.get_gradients(
            x_data, t_data, self.beta)

        self.mlp.weights_list = [w - self.eta * grad_w
                                 for w, grad_w in zip(self.mlp.weights_list, grad_w_list)]
        self.mlp.biases_list = [b - self.eta * grad_b
                                for b, grad_b in zip(self.mlp.biases_list, grad_b_list)]


class Momentum(Optimizer):
    
    def __init__(self, mlp, **kwargs):
        self.mlp = mlp

        self.eta = kwargs.pop("eta", 0.1)
        self.gamma = kwargs.pop("gamma", 0.9)
        self.beta = kwargs.pop("beta", 0)
        
        self.v_w_list = [np.zeros(w.shape) for w in mlp.weights_list]
        self.v_b_list = [np.zeros(b.shape) for b in mlp.biases_list]

    def process_batch(self, x_data, t_data):

        grad_w_list, grad_b_list = self.mlp.get_gradients(
            x_data, t_data, self.beta)

        self.v_w_list = [self.gamma * v_w + self.eta *
                         grad_w for v_w, grad_w in zip(self.v_w_list, grad_w_list)]
        self.v_b_list = [self.gamma * v_b + self.eta *
                         grad_b for v_b, grad_b in zip(self.v_b_list, grad_b_list)]

        self.mlp.weights_list = [
            w - v_w for w, v_w in zip(self.mlp.weights_list, self.v_w_list)]
        self.mlp.biases_list = [b - v_b for b,
                                v_b in zip(self.mlp.biases_list, self.v_b_list)]


class Nesterov(Optimizer):

    def __init__(self, mlp, **kwargs):
        self.mlp = mlp

        self.eta = kwargs.pop("eta", 0.1)
        self.gamma = kwargs.pop("gamma", 0.9)
        self.beta = kwargs.pop("beta", 0)
        
        self.v_w_list = [np.zeros(w.shape) for w in mlp.weights_list]
        self.v_b_list = [np.zeros(b.shape) for b in mlp.biases_list] 

    def process_batch(self, x_data, t_data):

        w_aux_list, b_aux_list = self.mlp.weights_list, self.mlp.biases_list

        self.mlp.weights_list = [w - self.gamma * v_w
                                 for w, v_w in zip(self.mlp.weights_list, self.v_w_list)]

        self.mlp.biases_list = [b - self.gamma * v_b
                                for b, v_b in zip(self.mlp.biases_list, self.v_b_list)]

        grad_w_list, grad_b_list = self.mlp.get_gradients(
            x_data, t_data, self.beta)

        self.v_w_list = [self.gamma * v_w + self.eta *
                         grad_w for v_w, grad_w in zip(self.v_w_list, grad_w_list)]
        self.v_b_list = [self.gamma * v_b + self.eta *
                         grad_b for v_b, grad_b in zip(self.v_b_list, grad_b_list)]

        self.mlp.weights_list = [
            w_aux - self.eta * v_w for w_aux, v_w in zip(w_aux_list, self.v_w_list)]
        self.mlp.biases_list = [b_aux - self.eta *
                                v_b for b_aux, v_b in zip(b_aux_list, self.v_b_list)]


class Adagrad(Optimizer):

    def __init__(self, mlp, **kwargs):
        self.mlp = mlp

        self.eta = kwargs.pop("eta", 0.1)
        self.beta = kwargs.pop("beta", 0)
        self.epsilon = kwargs.pop("epsilon", 1e-8)
        
        self.G_w_list = [np.ones(w.shape) for w in mlp.weights_list] #seguro que es ones y no zeros?
        self.G_b_list = [np.ones(b.shape) for b in mlp.biases_list]

    def process_batch(self, x_data, t_data):

        grad_w_list, grad_b_list = self.mlp.get_gradients(
            x_data, t_data, self.beta)

        self.G_w_list = [G_w + (grad_w ** 2) for G_w,
                         grad_w in zip(self.G_w_list, grad_w_list)]
        self.G_b_list = [G_b + (grad_b ** 2) for G_b,
                         grad_b in zip(self.G_b_list, grad_b_list)]

        self.mlp.weights_list = [w - (self.eta / np.sqrt(G_w + self.epsilon)) * grad_w for w, G_w, grad_w in
                                 zip(self.mlp.weights_list, self.G_w_list, grad_w_list)]

        self.mlp.biases_list = [b - (self.eta / np.sqrt(G_b + self.epsilon)) * grad_b for b, G_b, grad_b in
                                zip(self.mlp.biases_list, self.G_b_list, grad_b_list)]

class Adadelta(Optimizer):
    
    def init_aux_structures(self,mlp):
        self.avg_w_list = []
        self.avg_b_list = []
        
        self.avg_delta_w_list = []
        self.avg_delta_b_list = []

        for layer in range(mlp.nb_layers):
            new_avg_w = np.zeros((mlp.K_list[layer], mlp.K_list[layer + 1]))
            new_avg_b = np.zeros(mlp.K_list[layer + 1])
            self.avg_w_list.append(new_avg_w)
            self.avg_b_list.append(new_avg_b)
            new_avg_delta_w = (0.1**2)*np.ones((mlp.K_list[layer], mlp.K_list[layer + 1]))
            new_avg_delta_b = (0.1**2)*np.ones(mlp.K_list[layer + 1])
            self.avg_delta_w_list.append(new_avg_delta_w)
            self.avg_delta_b_list.append(new_avg_delta_b)
    
    
    def __init__(self, mlp, **kwargs):
        self.mlp = mlp

        self.beta = kwargs.pop("beta", 0)
        self.epsilon = kwargs.pop("epsilon", 1e-8)
        self.gamma = kwargs.pop("gamma", 0.9)
        
        self.init_aux_structures(mlp) 

    
    
    
    def process_batch(self, x_data, t_data):

        grad_w_list, grad_b_list = self.mlp.get_gradients(
            x_data, t_data, self.beta)
        
        self.avg_w_list = [self.gamma*avg_w + (1-self.gamma)*(grad_w**2) 
                            for avg_w, grad_w 
                            in zip(self.avg_w_list, grad_w_list)] 
        self.avg_b_list = [self.gamma*avg_b + (1-self.gamma)*(grad_b**2) 
                            for avg_b, grad_b 
                            in zip(self.avg_b_list, grad_b_list)]
        
        old_weights_list = self.mlp.weights_list
        
        self.mlp.weights_list = [w - (np.sqrt(avg_delta_w + self.epsilon) / np.sqrt(avg_w + self.epsilon)) * grad_w 
                                 for w, avg_delta_w, avg_w, grad_w in
                                 zip(self.mlp.weights_list, self.avg_delta_w_list, self.avg_w_list, grad_w_list)]
        
        delta_weights_list = [w_old - w_new for w_old, w_new in zip(old_weights_list,self.mlp.weights_list)]
        
        self.avg_delta_w_list = [self.gamma*avg_delta_w + (1-self.gamma)*(delta_w**2) for avg_delta_w, delta_w in
                                 zip(self.avg_w_list, delta_weights_list)]
        
        old_biases_list = self.mlp.biases_list
        
        self.mlp.biases_list = [b - (np.sqrt(avg_delta_b + self.epsilon) / np.sqrt(avg_b + self.epsilon)) * grad_b 
                                 for b, avg_delta_b, avg_b, grad_b in
                                 zip(self.mlp.biases_list, self.avg_delta_b_list, self.avg_b_list, grad_b_list)]
        
        delta_biases_list = [b_old - b_new for b_old, b_new in zip(old_biases_list,self.mlp.biases_list)]
        
        self.avg_delta_b_list = [self.gamma*avg_delta_b + (1-self.gamma)*(delta_b**2) for avg_delta_b, delta_b in
                                 zip(self.avg_b_list, delta_biases_list)]

class RMSprop(Optimizer):

    def __init__(self, mlp, **kwargs):
        self.mlp = mlp

        self.beta = kwargs.pop("beta", 0)
        self.eta = kwargs.pop("eta", 0.1)
        self.epsilon = kwargs.pop("epsilon", 1e-8)
        self.gamma = kwargs.pop("gamma", 0.9)
        
        self.avg_w_list = [np.zeros(w.shape) for w in mlp.weights_list]
        self.avg_b_list = [np.zeros(b.shape) for b in mlp.biases_list]
    
    def process_batch(self, x_data, t_data):

        grad_w_list, grad_b_list = self.mlp.get_gradients(
            x_data, t_data, self.beta)
        
        self.avg_w_list = [self.gamma*avg_w + (1-self.gamma)*(grad_w**2) 
                            for avg_w, grad_w 
                            in zip(self.avg_w_list, grad_w_list)] 
        self.avg_b_list = [self.gamma*avg_b + (1-self.gamma)*(grad_b**2) 
                            for avg_b, grad_b 
                            in zip(self.avg_b_list, grad_b_list)]
        
        
        self.mlp.weights_list = [w - (self.eta / np.sqrt(avg_w + self.epsilon)) * grad_w 
                                 for w, avg_w, grad_w in
                                 zip(self.mlp.weights_list, self.avg_w_list, grad_w_list)]
        
        
        self.mlp.biases_list = [b - (self.eta/ np.sqrt(avg_b + self.epsilon)) * grad_b 
                                 for b, avg_b, grad_b in
                                 zip(self.mlp.biases_list, self.avg_b_list, grad_b_list)]
        
       
class Adam(Optimizer):

    def __init__(self, mlp, **kwargs):
        self.mlp = mlp

        self.beta = kwargs.pop("beta", 0)
        self.eta = kwargs.pop("eta", 0.1)
        self.epsilon = kwargs.pop("epsilon", 1e-8)
        self.beta_1 = kwargs.pop("beta_1", 0.9)
        self.beta_2 = kwargs.pop("beta_2", 0.999)
        
        self.v_w_list = [np.zeros(w.shape) for w in mlp.weights_list]
        self.v_b_list = [np.zeros(b.shape) for b in mlp.biases_list]
        self.m_w_list = [np.zeros(w.shape) for w in mlp.weights_list]
        self.m_b_list = [np.zeros(b.shape) for b in mlp.biases_list]

    
    def process_batch(self, x_data, t_data):

        grad_w_list, grad_b_list = self.mlp.get_gradients(
            x_data, t_data, self.beta)
        
        self.v_w_list = [self.beta_2*v_w + (1-self.beta_2)*(grad_w**2) 
                            for v_w, grad_w 
                            in zip(self.v_w_list, grad_w_list)] 
        
        # self.v_w_list = [ v_w / (1 - self.beta_2) for v_w in self.v_w_list]
        
        self.v_b_list = [self.beta_2*v_b + (1-self.beta_2)*(grad_b**2) 
                            for v_b, grad_b 
                            in zip(self.v_b_list, grad_b_list)]
        
        # self.v_b_list = [ v_b / (1 - self.beta_2) for v_b in self.v_b_list]
        
        
        self.m_w_list = [self.beta_1*m_w + (1-self.beta_1)*(grad_w) 
                            for m_w, grad_w 
                            in zip(self.m_w_list, grad_w_list)] 
        
        # self.m_w_list = [ m_w / (1 - self.beta_1) for m_w in self.m_w_list]
        
        self.m_b_list = [self.beta_1*m_b + (1-self.beta_1)*(grad_b) 
                            for m_b, grad_b 
                            in zip(self.m_b_list, grad_b_list)]
        
        # self.m_b_list = [ m_b / (1 - self.beta_1) for m_b in self.m_b_list]
        
        
        self.mlp.weights_list = [w - (self.eta / (np.sqrt(v_w) + self.epsilon)) * m_w 
                                 for w, v_w, m_w in
                                 zip(self.mlp.weights_list, self.v_w_list, self.m_w_list)]
        
        
        self.mlp.biases_list = [b - (self.eta / (np.sqrt(v_b) + self.epsilon)) * m_b 
                                 for b, v_b, m_b in
                                 zip(self.mlp.biases_list, self.v_b_list, self.m_b_list)]
