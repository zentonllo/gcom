# -*- coding: utf-8 -*-
"""Module modeling a Multilayer Perceptron, featuring a tiny example

Example
-------

    Executing this script will run a test consisting on training two linearly
    separable set of points

    $ python mlp.py

    On the other hand, we can import this module and instanciating an Multi-
    layer Perceptron in the following way:

        mlp = MLP(K_list, activation_functions, diff_activation_functions)

    Then, we train the MLP in this fashion:

        mlp.train(x_data, t_data,
              epochs=1000, batch_size=20, initialize_weights=False,
              method='adam', eta=0.1, beta=0, gamma=0.9, beta_1=0.9,
              beta_2=0.999, epsilon=1e-8, print_cost=True)

    These arguments are widely explained in the MLP train method

"""
from __future__ import division, print_function

import sys
import numpy as np
import mlpOptimizer as mlpo

__author__ = "Ignacio Casso, Daniel Gamo, Gwydion J. Martín, Alberto Terceño"


class MLP(object):
    """Class that models a Multilayer Perceptron

    Here they are some notation and assumptions that have been made throughout
    this module:

        - N: input data examples
        - R: number of layers (the input layer is not considered, and it will
                               be named as the 0-layer since it doesn't have
                               activation functions nor weights matrix)
        - Dk: number of neurons on k-layer

        - A weights matrix on each layer has dimension (Dk, Dk+1). An element
          placed in the i-th row, j-th columns can be seen as the i-th weight
          of the j-th neuron on layer k+1. Hence, units will multiply the
          weights matrixes by their left hand side
        - The matrix which groups the N different input data examples places
          each one of them in rows.
        - Weights and biases matrixes for the k-th layer can be found in the
          (k-1)-th index of the lists which hold all these matrixes. It is
          important to keep this gap in mind

    Attributes
    ----------
    K_list : [int]
        List containing (in order) the number of neurons on each layer (including the 
        input and the output layer)
    nb_layers : int
        Number of layers in the neuronal networks (excluding the input one)
    activation_functions : [function]
        List of the activation functions used on each layer
    diff_activation_functions : [function] 
        List holding the derivatives functions of the corresponding 
        activation ones used on each layer
    init_seed : int 
        Seed used in order to initialize the weights    
    weights_list : [np.array]
        List which holds in its (k-1)-th index the weights matrix corres-
        ponding to the k-th layer
    biases_list : [np.array]
        List which holds in its (k-1)-th index the bias vector corres-
        ponding to the k-th layer
    y : [np.array]
        Multilayer Perceptron outputs for certain input data 
    """

    def __init__(self, K_list,
                 activation_functions, diff_activation_functions,
                 init_seed=None):
        """Example of docstring on the __init__ method.

        The __init__ method may be documented in either the class level
        docstring, or as a docstring on the __init__ method itself.

        Either form is acceptable, but the two should not be mixed. Choose one
        convention to document the __init__ method and be consistent with it.

        Note
        ----
        Do not include the `self` parameter in the ``Parameters`` section.

        Parameters
        ----------
        param1 : str
            Description of `param1`.
        param2 : list(str)
            Description of `param2`. Multiple
            lines are supported.
        param3 : :obj:`int`, optional
            Description of `param3`.

        """
        self.K_list = K_list
        self.nb_layers = len(K_list) - 1  # = R

        # We suppose they're lists of R elements
        self.activation_functions = activation_functions
        # and that the k-th index represents the (k+1)-th layer
        self.diff_activation_functions = diff_activation_functions

        self.init_seed = init_seed

        self.weights_list = None  # list of R (Dk,Dk+1) matrix
        self.biases_list = None  # list of R row vectors of Dk+1 elements

        self.y = None  # (N,Dr) matrix

        self.init_weights()

# %% definition of activation functions and derivatives

    #@staticmethod
    # def sigmoid(z):
     # return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (np.exp(z) +
     # 1))

    @staticmethod
    def sigmoid(z):
        y = np.zeros(z.shape)
        masc1 = z >= 0
        masc2 = z < 0
        y[masc1] = 1 / (1 + np.exp(-z[masc1]))
        y[masc2] = np.exp(z[masc2]) / (np.exp(z[masc2]) + 1)
        return y

    @staticmethod
    def dsigmoid(z):
        return MLP.sigmoid(z) * (1 - MLP.sigmoid(z))

    @staticmethod
    def dtanh(z):
        return 1 - np.tanh(z)**2

    @staticmethod
    def relu(z):
        # He leido que z * (z > 0) es lo más rápido para la relu
        return np.maximum(z, 0)

    @staticmethod
    def drelu(z):
        # drelu(0)=1 by agreement
        return np.where(z > 0, 1, 0)

    @staticmethod
    def identity(z):
        return z

    @staticmethod
    def didentity(z):  # it only works with numpy arrays
        return np.ones(z.shape)

    @staticmethod
    def softmax(z):
        max_values = np.amax(z, axis=1).reshape(z.shape[0], 1)
        x = z - max_values
        sum_exp = np.sum(np.exp(x), axis=1).reshape(z.shape[0], 1)
        return np.exp(x) / sum_exp

    # %% cost functions
    @staticmethod
    def binary_cross_entropy(y, t_data):
        x = np.maximum(y, 10**-15)
        return -np.sum(t_data * np.log(x) + (1 - t_data) * np.log(1 - x))
        # return -np.sum(t_data * np.log(y) + (1 - t_data) * np.log(1 - y))

    @staticmethod
    def softmax_cross_entropy(y, t_data):
        x = np.maximum(y, 10**-15)
        return -np.sum(t_data * np.log(x))
        # return -np.sum(t_data * np.log(y))

    @staticmethod
    def cost_L2(y, t_data):
        return 0.5 * np.sum((y - t_data)**2)

    # %% simple weights initialization

    def init_weights(self):

        if self.init_seed:
            np.random.seed(self.init_seed)

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
    def get_activations_and_units(self, x, wb=None):
        """Class methods are similar to regular functions.

        Note
        ----
        Do not include the `self` parameter in the ``Parameters`` section.

        Parameters
        ----------
        param1
            The first parameter.
        param2
            The second parameter.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        if wb is None:
            weights_list, biases_list = self.weights_list, self.biases_list
        else:
            weights_list, biases_list = wb

        activations = [x]
        units = [x]
        z = x
        for i in range(self.nb_layers):
            # matrix + row vector, so it adds the vector to each of the matrix
            # rows
            a = z.dot(weights_list[i]) + biases_list[i]
            activations.append(a)
            z = self.activation_functions[i](a)
            units.append(z)

        self.y = z

        return activations, units

    # %% backpropagation
    # This function calculates the error gradient for each of the data and
    # averages them. All the gradients are calculated at the same time using
    # (N,?) matrix instead of vectors.
    # We use : x = (N,D0) matrix, t = (N,Dr) matrix, delta_k = (N,Dk) matrix
    def get_gradients(self, x, t, beta=0, wb=None):

        # Slightly different from the class notes due to the separation of bs
        # and Ws and the change of the index to name the weights.
        # The functions returns a list of shifted index (k-th index = (k+1)-th
        # layer gradients; the layer 0 (input) has no Ws)

        if wb is None:
            weights_list, biases_list = self.weights_list, self.biases_list
        else:
            weights_list, biases_list = wb

        activations, units = self.get_activations_and_units(x, wb)

        N = x.shape[0]
        grad_w_list = [0] * self.nb_layers
        grad_b_list = [0] * self.nb_layers

        delta_k1 = None  # delta value for the next layer

        ks = range(1, self.nb_layers + 1)
        ks.reverse()
        for k in ks:  # r, ..., 1

            # we calculate the new delta values
            if (k < self.nb_layers):
                # weights of the (k+1)-th layer
                w = weights_list[k]
                # activation function derivative on layer k
                dh = self.diff_activation_functions[k - 1]
                # activations from layer k
                a = activations[k]
                delta_k = (delta_k1.dot(w.T)) * dh(a)
            else:
                # we can assume the derivative of En respect to the last
                # activations layer is y-t
                delta_k = units[k] - t

            grad_wk = (np.einsum(
                'ij,ik', units[k - 1], delta_k) / N) + (beta * weights_list[k - 1])
            grad_w_list[k - 1] = grad_wk

            grad_bk = np.sum(delta_k, axis=0) / N
            grad_b_list[k - 1] = grad_bk

            delta_k1 = delta_k

        ##

        return grad_w_list, grad_b_list

    # %%
    # training method for the neuron
    def train(self, x_data, t_data, epochs, batch_size,
              initialize_weights=False, print_cost=False,
              **method_args):

        opt = mlpo.Optimizer.get_optimizer(self, **method_args)

        if initialize_weights:
            self.init_weights()

        nb_data = x_data.shape[0]
        index_list = np.arange(nb_data)
        nb_batches = int(nb_data / batch_size)

        for _ in range(epochs):
            np.random.shuffle(index_list)
            for batch in range(nb_batches):
                indexes = index_list[batch *
                                     batch_size:(batch + 1) * batch_size]
                opt.process_batch(x_data[indexes], t_data[indexes])

            if print_cost:
                x_batch = x_data
                t_batch = t_data
                self.get_activations_and_units(x_batch)
                if self.activation_functions[-1] == MLP.sigmoid:
                    sys.stdout.write('cost = %f\r' %
                                     MLP.binary_cross_entropy(self.y, t_batch))
                    sys.stdout.flush()
                elif self.activation_functions[-1] == MLP.softmax:
                    sys.stdout.write('cost = %f\r' %
                                     MLP.softmax_cross_entropy(
                                         self.y, t_batch))
                    sys.stdout.flush()
                else:
                    sys.stdout.write('cost = %f\r' %
                                     MLP.cost_L2(self.y, t_batch))
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
    t_data = np.asarray([0] * nb_black + [1] * nb_red).reshape(nb_data, 1)

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
              epochs=1000, batch_size=20, initialize_weights=False, method='adam', eta=0.1,
              beta=0, gamma=0.9, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
              print_cost=True)
