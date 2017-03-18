#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 13:13:13 2017

@author: avaldes
"""

from __future__ import division, print_function


import sys
import numpy as np


class MLP(object):

    # Algunos comentarios sobre notacion y estructura de los vectores y matrices
    # N = numero de datos
    # R = numero de capas sin contar la capa imput(que no tiene pesos ni funciones de activacion)
    # se usaran subindices para denotar estas capas, siendo 0 el subindice de la capa input
    # Dk = numero de neuronas en la capa k

    # La matriz W de pesos de cada capa tendra dimension (Dk, Dk+1) (Wij es el peso i-esimo de la
    # neurona j-esima de la capa k+1. Esta decision viene forzada por el
    # template, e implica que:

        # para operar un vector de units sobre ella hay que multiplicarlo por la izquierda, y entonces
        # tanto las activaciones como las units seran vectores fila.

        # Las matrices que agrupen vectores de N datos diferentes (como la matriz x o la y) tendran
        # una fila para cada dato, es decir, tendran dimension (N,?)

    # Las listas de matrices de pesos y biases tienen los correspondientes a la capa k-esima en el
    # indice k-1. Hay que tener cuidado con este desfase

    # self.nb_layers = R

    def __init__(self, K_list,
                 activation_functions, diff_activation_functions,
                 init_seed=None):

        self.K_list = K_list
        self.nb_layers = len(K_list) - 1  # = R

        # Suponemos que son listas de longitud R, y al indice k
        self.activation_functions = activation_functions
        # le corresponde la capa k+1
        self.diff_activation_functions = diff_activation_functions

        self.init_seed = init_seed

        self.weights_list = None  # lista de R matrices (Dk,Dk+1)
        self.biases_list = None  # lista de R vectores fila de dimension Dk+1

        self.grad_w_list = None  # lista de R matrices (Dk,Dk+1)
        self.grad_b_list = None  # lista de R vectores fila de dimenison Dk+1

        self.activations = None  # lista de R+1 matrices (N,Dk)
        self.units = None  # lista de R+1 matrices (N, Dk)
        self.y = None  # matriz (N, Dr)

        self.init_weights()

# %% definition of activation functions and derivatives
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def dsigmoid(z):
        return MLP.sigmoid(z) * (1 - MLP.sigmoid(z))

    @staticmethod
    def dtanh(z):
        return 1 - np.tanh(z)**2

    @staticmethod
    def relu(z):
        return np.maximum(z, 0)

    @staticmethod
    def drelu(z):
        z[z >= 0] = 1   # drelu(0)=1 por convenio
        z[z < 0] = 0
        return z

    @staticmethod
    def identity(z):
        return z

    @staticmethod
    def didentity(z): # Solo funciona para numpy arrays
        return [1] * z.shape[0]

    @staticmethod
    def softmax(z):
        sum_exp = np.sum(np.exp(z))
        return np.exp(z) / sum_exp

    # %% cost functions
    @staticmethod
    def binary_cross_entropy(y, t_data):
        return -np.sum(t_data * np.log(y) + (1 - t_data) * np.log(1 - y),
                       axis=0) #no hace falta axis=0? Solo se va a llamar para datos individuales?

    @staticmethod
    def softmax_cross_entropy(y, t_data):
        return -np.sum(t_data * np.log(y), axis=0)

    @staticmethod
    def cost_L2(y, t_data):
        return 0.5*np.sum((y - t_data)**2)

    # %% simple weights initialization

    def init_weights(self):

        if self.init_seed:
            np.random.seed(self.seed)

        weights_list = []
        biases_list = []

        for layer in range(self.nb_layers):
            new_W = np.random.randn(self.K_list[layer], self.K_list[layer + 1])
            new_b = np.zeros(self.K_list[layer + 1])
            weights_list.append(new_W)
            biases_list.append(new_b)

        self.weights_list = weights_list
        self.biases_list = biases_list

    # %% feed forward pass
    # x matrixz(N,D0)
    def get_activations_and_units(self, x):

        activations = [x]
        units = [x]
        z = x
        for i in range(self.nb_layers):
            # your code here
            a = z.dot(self.weights_list[i]) + self.biases_list[i] #matriz + vector fila, pero funciona (suma el vector a cada fila de la matriz)
            activations.append(a)
            z = self.activation_functions[i](a)
            units.append(z)

        self.activations = activations
        self.units = units
        self.y = z

    # %% backpropagation 
    # Calcula el gradiente del error para cada dato y promedia.
    # Todos los gradientes se calculan a la vez usando matrices (N,?) en vec de vectores. Para ello usamos:
    # x matriz(N,D0), t matriz(N,Dr), delta_k matriz(N,Dk)
    def get_gradients(self, x, t, beta=0):

        # Ligeramente distinto a las notas de clase debido a la separacion de las bs y los ws
        # y al cambio de indices para denotar los pesos

        # Debe devolver una lista con los indices desfasados como weights_list (indice k = gradientes de la capa k+1, la capa 0 (input) no tiene Ws)

        self.get_activations_and_units(x)

        N = x.shape[0]
        grad_w_list = [0]*self.nb_layers
        grad_b_list = [0]*self.nb_layers

        # your code here
        delta_k1 = None #delta de la capa siguiente. ¿Hace falta declararlo en python para poder ejecutar la ultima instruccion del for?

        ks = range(1, self.nb_layers+1)
        ks.reverse()
        for k in ks:  # r, ..., 1

            #Calculamos los nuevos deltas
            if (k<self.nb_layers):
                w = self.weights_list[k]  # pesos de la capa k + 1
                dh = self.diff_activation_functions[k-1]  # derivada de la funcion de activacion de la capa k
                a = self.activations[k]  # activaciones de la capa k
                delta_k = (delta_k1.dot(w.T))*dh(a)
            else:
                delta_k = self.y - t #podemos asumir que la derivada de En respecto de la ultima capa de activaciones es y-t

            grad_wk = MLP.get_w_gradients(self.units[k-1], delta_k) + beta*self.weights_list[k-1]
            grad_w_list[k-1] = grad_wk

            grad_bk = np.sum(delta_k, axis=0)/N + beta*self.biases_list[k-1]
            grad_b_list[k-1] = grad_bk

            delta_k1 = delta_k

        ##

        self.grad_w_list = grad_w_list
        self.grad_b_list = grad_b_list

    @staticmethod
    # z matriz (N,D), delta matriz (N,D')
    # intentar usar funciones de numpy como einsum para hacerlo mas corto y eficiente
    # hay que devolver la suma promediada de las N matrices (DxD') resultantes de multiplicar matricialmente
    # (k-esima fila de z, transpuesta)*(k-esima fila de delta), para cada k de 1 a N
    def get_w_gradients(z, delta):
        N = z.shape[0]
        sum_grads = np.zeros((z.shape[1], delta.shape[1]))

        for k in range(N):
            grad = np.zeros((z.shape[1], delta.shape[1]))
            # Yo haría grad = z[k].T.dot(delta[k]), sum_grads =sum_grads + grad y nos ahorramos el for interno
            for i in range(z.shape[1]):
                grad[i] = z[k,i]*delta[k]
            sum_grads = sum_grads + grad

        return sum_grads/N

    # %% 
    def train(self, x_data, t_data,
              epochs, batch_size,
              initialize_weights=False,
              epsilon=0.01,
              beta=0,
              print_cost=False):
  
        if initialize_weights:
            self.init_weights()

        nb_data = x_data.shape[0]
        index_list = np.arange(nb_data)
        nb_batches = int(nb_data / batch_size)

        for _ in range(epochs):
            np.random.shuffle(index_list)
            for batch in range(nb_batches):
                # your code here
                indexes = index_list[batch*batch_size:(batch+1)*batch_size]
                self.get_gradients(x_data[indexes], t_data[indexes], beta)
                self.weights_list = [self.weights_list[k] - epsilon*self.grad_w_list[k] for k in range(self.nb_layers)]
                self.biases_list = [self.biases_list[k] - epsilon*self.grad_b_list[k] for k in range(self.nb_layers)]
                

            if print_cost:
                x_batch = x_data
                t_batch = t_data
                self.get_activations_and_units(x_batch)
                if self.activation_functions[-1] == MLP.sigmoid:
                    sys.stdout.write('cost = %f\r' %MLP.binary_cross_entropy(self.y, t_batch))
                    sys.stdout.flush()
                elif self.activation_functions[-1] == MLP.softmax:
                    sys.stdout.write('cost = %f\r' %MLP.softmax_cross_entropy(self.y, t_batch))
                    sys.stdout.flush()
                else:
                    sys.stdout.write('cost = %f\r' %MLP.cost_L2(self.y, t_batch))
                    sys.stdout.flush()

# %% let's experiment

if __name__ == '__main__':

    # %% Create data
    # np.random.seed(5)
    nb_black = 50
    nb_red = 50
    nb_data = nb_black + nb_red
    x_data_black = np.random.randn(nb_black, 2) + np.array([0, 0])
    x_data_red = np.random.randn(nb_red, 2) + np.array([10, 10])

    x_data = np.vstack((x_data_black, x_data_red))
    t_data = np.asarray([0]*nb_black + [1]*nb_red).reshape(nb_data, 1)

# %% Net structure
    D = x_data.shape[1]  # initial dimension
    K = 1  # final dimension

    K_list = [D, K]  # list of dimensions

    activation_functions = [MLP.sigmoid]
    diff_activation_functions = [MLP.dsigmoid]


# %%
    mlp = MLP(K_list, activation_functions, diff_activation_functions)
    

# %% Train begins
    mlp.train(x_data, t_data,
              epochs=1000, batch_size=10, initialize_weights=True, epsilon=0.1, print_cost=True)
