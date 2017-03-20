# -*- coding: utf-8 -*-

from __future__ import division, print_function

import sys
import numpy as np

__author__ = "Ignacio Casso, Daniel Gamo, Gwydion J. Martín, Alberto Terceño"


class MLP(object):

    # Here are some appreciations about notation and the structure of vectors and matrix
    # N = data number
    # R = layers number (without the imput layer, as it has no activation functions nor weights
    # subindex will be used to name these layers, with 0 being the input layer
    # Dk = number of neurons on layer k

    # The weights' matrix W for each layer will have dimension (Dk, Dk+1) (Wij es el peso i-esimo de la
    # Wij is the i-th weight of the j-th neuron on layer k+1. This decision is forced because of the template, so:

        # to operate with a units vector on the matrix, you have to multiply by the left, y entonces
        # and so both the activations and the units will be raw vectors.

        # the matrix that group vectors with N different data (like the matrix x or y)
        # will have a raw for each data, so they'll have dimension (N,?).

    # The lists of weights' and biases' matrix have the k-th layer data in the (k-1)-th index.
    # It's important to keep this phase shift in mind

    # self.nb_layers = R

    def __init__(self, K_list,
                 activation_functions, diff_activation_functions,
                 init_seed=None):

        self.K_list = K_list
        self.nb_layers = len(K_list) - 1  # = R

        # We suppose they're lists of R elements
        self.activation_functions = activation_functions
        # and that the k-th index represents the (k+1)-th layer
        self.diff_activation_functions = diff_activation_functions

        self.init_seed = init_seed

        self.weights_list = None  # list of R (Dk,Dk+1) matrix
        self.biases_list = None  # list of R raw vectors of Dk+1 elements

        self.grad_w_list = None  # list of R (Dk,Dk+1) matrix 
        self.grad_b_list = None  # list of R raw vectors of Dk+1 elements

        self.activations = None  # list of R+1 (N,Dk) matrix
        self.units = None  # list of R+1 (N,Dk) matrix
        self.y = None  # (N,Dr) matrix

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
        z[z >= 0] = 1   # drelu(0)=1 by agreement
        z[z < 0] = 0
        return z

    @staticmethod
    def identity(z):
        return z

    @staticmethod
    def didentity(z): # it only works with numpy arrays
        return [1] * z.shape[0]

    @staticmethod
    def softmax(z):
        sum_exp = np.sum(np.exp(z))
        return np.exp(z) / sum_exp

    # %% cost functions
    @staticmethod
    def binary_cross_entropy(y, t_data):
        return -np.sum(t_data * np.log(y) + (1 - t_data) * np.log(1 - y),
                       axis=0) 

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
    # x = (N,D0) matrix
    def get_activations_and_units(self, x):

        activations = [x]
        units = [x]
        z = x
        for i in range(self.nb_layers):
            # matrix + raw vector, so it adds the vector to each of the matrix' raws
            a = z.dot(self.weights_list[i]) + self.biases_list[i] 
            activations.append(a)
            z = self.activation_functions[i](a)
            units.append(z)

        self.activations = activations
        self.units = units
        self.y = z

    # %% backpropagation 
    # This function calculates the error gradient for each of the data and averages them.
    # All the gradients are calculated at the same time using (N,?) matrix instead of vectors. 
    # We use : x = (N,D0) matrix, t = (N,Dr) matrix, delta_k = (N,Dk) matrix
    def get_gradients(self, x, t, beta=0):

        # Slightly different from the class notes due to the separation of bs and Ws
        # and the change of the index to name the weights.
        # The functions returns a list of shifted index (k-th index = (k+1)-th layer gradients; the layer 0 (input) has no Ws)

        self.get_activations_and_units(x)

        N = x.shape[0]
        grad_w_list = [0]*self.nb_layers
        grad_b_list = [0]*self.nb_layers

        delta_k1 = None # delta value for the next layer. ¿Hace falta declararlo en python para poder ejecutar la ultima instruccion del for?

        ks = range(1, self.nb_layers+1)
        ks.reverse()
        for k in ks:  # r, ..., 1

            # we calculate the new delta values
            if (k<self.nb_layers):
                w = self.weights_list[k]  # weights of the (k+1)-th layer
                dh = self.diff_activation_functions[k-1]  # derived from the activation function on layer k
                a = self.activations[k]  # activations from layer k
                delta_k = (delta_k1.dot(w.T))*dh(a)
            else:
                delta_k = self.y - t # we can assume the derived from En with respect to the last activations layer is y-t

            grad_wk = MLP.get_w_gradients(self.units[k-1], delta_k) + beta*self.weights_list[k-1]
            grad_w_list[k-1] = grad_wk

            grad_bk = np.sum(delta_k, axis=0)/N + beta*self.biases_list[k-1]
            grad_b_list[k-1] = grad_bk

            delta_k1 = delta_k

        ##

        self.grad_w_list = grad_w_list
        self.grad_b_list = grad_b_list

    @staticmethod
    # z = (N,D) matrix, delta = (N,D') matrix
    # the function returns the average sum of the N (D,D') matrix result of multiplying
    # (k-th raw from z, transposed)*(k-th raw from delta), for each k from 1 to N
    def get_w_gradients(z, delta):
        N = z.shape[0]
        sum_grads = np.zeros((z.shape[1], delta.shape[1]))

        for k in range(N):
            grad = np.zeros((z.shape[1], delta.shape[1]))
            grad = z[k].T.dot(delta[k])
            sum_grads = sum_grads + grad

        return sum_grads/N

    # %% 
    # training method for the neuron
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
