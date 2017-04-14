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

    These arguments are widely explained in the MLP 'train' method

"""
from __future__ import division, print_function

import sys
import numpy as np
import mlpOptimizer as mlpo

__author__ = "Ignacio Casso, Daniel Gamo, Gwydion J. Martín, Alberto Terceño"


class MLP(object):
    """Class that models a Multilayer Perceptron (MLP)

    Here they are some notation and assumptions that have been made throughout
    this module:

        - N: number of input data examples
        - R: number of layers (the input layer is not considered, and it will
                               be named as the 0-layer since it doesn't have
                               activation functions nor weights matrix)
        - Dk: number of neurons on k-layer

        - A weights matrix on each layer has dimension (Dk, Dk+1). An element
          placed in the i-th row, j-th columns can be seen as the i-th weight
          of the j-th neuron on layer k+1. Hence, units will multiply the
          weights matrixes by their left hand side
        - The matrix which groups the N different input data examples places
          each one of them in rows. Matrix containing the activations and units
          for each layer will be computed and saved in the same way
        - Weights and biases matrixes for the k-th layer can be found in the
          (k-1)-th index of the lists which hold all these matrixes. It is
          important to keep this gap in mind

    Attributes
    ----------
    K_list : [int]
        List containing (in order) the number of neurons on each layer
        (including the input and the output layer)
    nb_layers : int
        Number of layers in the neuronal networks (excluding the input one)
    activation_functions : [function]
        Ordered list of the activation functions used on each layer
    diff_activation_functions : [function]
        Ordered list holding the derivatives functions of the corresponding
        activation ones used on each layer
    init_seed : int
        Seed used in order to initialize weights
    weights_list : np.array
        List which holds in its (k-1)-th index the weights matrix corres-
        ponding to the k-th layer
    biases_list : np.array
        List which holds in its (k-1)-th index the bias vector corres-
        ponding to the k-th layer
    y : [np.array]
        Multilayer Perceptron outputs
    reg_method : string
        Indicates the regularization method to be used. There have been
        implemented: 'L1', 'L2' and 'Elastic_Net' regularizations
    beta : int
        Regularization parameter


    """

    def __init__(self, K_list,
                 activation_functions, diff_activation_functions,
                 init_seed=None):
        """__init__ method for the MLP class

        Sets up hyperparameters for the MLP class

        Parameters
        ----------
        K_list : [int]
             List containing (in order) the number of neurons on each layer
             (including the input and the output layer)
        activation_functions : [function]
             Ordered list of the activation functions used on each layer
        diff_activation_functions : [function]
             Ordered list holding the derivatives functions of the
             corresponding activation ones used on each layer
        init_seed : int
            Seed used in order to initialize weights

        """
        self.K_list = K_list
        self.nb_layers = len(K_list) - 1

        self.activation_functions = activation_functions
        self.diff_activation_functions = diff_activation_functions

        self.init_seed = init_seed

        self.weights_list = None
        self.biases_list = None

        # At the beginning there is no input yet
        self.y = None

        # Regularization parameters will be set afterwards
        self.beta = None
        self.reg_method = None

        # Initialize weights when instatiating a MLP object
        self.init_weights()

# %% definition of activation functions and derivatives

    @staticmethod
    def sigmoid(z):
        """Numerically stable implementation of the sigmoid function


        Parameters
        ----------
        z : np.array
             Matrix containing activations for each data sample. The activation
             for the i-th data sample is stored in the i-th row, as we have
             assumed

        Returns
        -------
        np.array
            Element-wise sigmoid function applied to parameter z

        """
        y = np.zeros(z.shape)
        masc1 = z >= 0
        masc2 = z < 0
        y[masc1] = 1 / (1 + np.exp(-z[masc1]))
        y[masc2] = np.exp(z[masc2]) / (np.exp(z[masc2]) + 1)
        return y

    @staticmethod
    def dsigmoid(z):
        """Implementation of the derivative sigmoid function


        Parameters
        ----------
        z : np.array
             Matrix containing activations for each data sample. The activation
             for the i-th data sample is stored in the i-th row, as we have
             assumed

        Returns
        -------
        np.array
            Element-wise derivative sigmoid function applied to parameter z

        """
        return MLP.sigmoid(z) * (1 - MLP.sigmoid(z))

    @staticmethod
    def dtanh(z):
        """Implementation of the derivative hyperbolic tangent function


        Parameters
        ----------
        z : np.array
             Matrix containing activations for each data sample. The activation
             for the i-th data sample is stored in the i-th row, as we have
             assumed

        Returns
        -------
        np.array
            Element-wise derivative hyperbolic tangent function applied
            to parameter z

        """
        return 1 - np.tanh(z)**2

    @staticmethod
    def relu(z):
        """Implementation of the rectifier activation function


        Parameters
        ----------
        z : np.array
             Matrix containing activations for each data sample. The activation
             for the i-th data sample is stored in the i-th row, as we have
             assumed

        Notes
        ----------
        This implementation has been choosen since it is been proved to be a
        fast way

        Source: https://goo.gl/QIiHFP


        Returns
        -------
        np.array
            Element-wise rectifier activation function applied to parameter z

        """
        return z * (z > 0)
        # return np.maximum(z, 0)

    @staticmethod
    def drelu(z):
        """Implementation of the derivative rectifier function


        Parameters
        ----------
        z : np.array
             Matrix containing activations for each data sample. The activation
             for the i-th data sample is stored in the i-th row, as we have
             assumed

        Notes
        ----------
        We have decided to define drelu(0) = 0

        Returns
        -------
        np.array
            Element-wise derivative rectifier function applied to parameter z

        """

        return np.where(z > 0, 1, 0)

    @staticmethod
    def identity(z):
        """Implementation of the identitity function

        Parameters
        ----------
        z : np.array
             Matrix containing activations for each data sample. The activation
             for the i-th data sample is stored in the i-th row, as we have
             assumed

        Returns
        -------
        np.array
            Element-wise identity function applied to parameter z

        """
        return z

    @staticmethod
    def didentity(z):
        """Implementation of the derivative identitity function

        Parameters
        ----------
        z : np.array
             Matrix containing activations for each data sample. The activation
             for the i-th data sample is stored in the i-th row, as we have
             assumed

        Notes
        ----------
        This method only works with numpy arrays

        Returns
        -------
        np.array
            Matrix filled with ones (ie, derivative identity function). It has
            the exact shape than the input parameter

        """
        return np.ones(z.shape)

    @staticmethod
    def softmax(z):
        """Numerically stable implementation of the softmax function

        Parameters
        ----------
        z : np.array
             Matrix containing activations for each data sample. The activation
             for the i-th data sample is stored in the i-th row, as we have
             assumed

        Returns
        -------
        np.array
            Row-wise softmax function applied to parameter z

        """
        max_values = np.amax(z, axis=1).reshape(z.shape[0], 1)
        x = z - max_values
        sum_exp = np.sum(np.exp(x), axis=1).reshape(z.shape[0], 1)
        return np.exp(x) / sum_exp

# %% cost functions

    @staticmethod
    def binary_cross_entropy(y, t_data):
        """Numerically stable implementation of the Binary Cross entropy
           cost function

        Parameters
        ----------
        y : np.array
            (N,Dr) matrix which contains the Multilayer Perceptron outputs
            for the input data samples labeled with t_data

        t_data : np.array
            (N,Dr) matrix representing labels for each data sample. If these
            labels correspond to a binary classification problems (Dr = 1),
            there are as many labels as input data samples. On the other hand,
            if we have a multiclass classification problem (Dr > 1),
            labels for each sample are row-wise stacked


        Returns
        -------
        int
            Binary cross entropy function applied to the MLP outputs and labels

        """
        x = np.maximum(y, 10**-15)
        return -np.sum(t_data * np.log(x) + (1 - t_data) * np.log(1 - x))

    @staticmethod
    def softmax_cross_entropy(y, t_data):
        """Numerically stable implementation of the Softmax Cross entropy
           cost function

        Parameters
        ----------
        y : np.array
            (N,Dr) matrix which contains the Multilayer Perceptron outputs
            for the input data samples labeled with t_data

        t_data : np.array
            (N,Dr) matrix representing labels for each data sample. If these
            labels correspond to a binary classification problems (Dr = 1),
            there are as many labels as input data samples. On the other hand,
            if we have a multiclass classification problem (Dr > 1),
            labels for each sample are row-wise stacked


        Returns
        -------
        int
           Softmax cross entropy function applied to the MLP outputs and labels

        """
        x = np.maximum(y, 10**-15)
        return -np.sum(t_data * np.log(x))

    @staticmethod
    def cost_L2(y, t_data):
        """Implementation of the sum squared error cost function

        Notes
        ----------
        Preferably used for one-variable function regression problems

        Parameters
        ----------
        y : np.array
            (N,Dr) matrix which contains the Multilayer Perceptron outputs
            for the input data samples labeled with t_data

        t_data : np.array
            (N,Dr) matrix representing labels for each data sample. If these
            labels correspond to a binary classification problems (Dr = 1),
            there are as many labels as input data samples. On the other hand,
            if we have a multiclass classification problem (Dr > 1),
            labels for each sample are row-wise stacked


        Returns
        -------
        int
            Sum squared error cost function applied to the MLP outputs and
            labels

        """
        return 0.5 * np.sum((y - t_data)**2)

# %% simple weights initialization

    def init_weights(self):
        """Random weight initialization

        Initialize random weights and biases for the Multilayer Perceptron.
        If the initial seed has been set then it is used for comparing perfor-
        mances with another multilayer perceptrons.


        Returns
        -------
        None

        """
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

    def get_activations_and_units(self, x, wb=None):
        """Computes activations and units on each layer until it gets the
           final outputs


        Parameters
        ----------
        x : np.array
            (N,D0) matrix holding each input data sample. The i-th data sample
            is stored in the i-th row.
        wb : tuple
            Tuple consisting of a weights and a biases list to be used for
            computing the result. If this parameter is not passed then the MLP
            own weights are used for the computations

        Returns
        -------
        Tuple
            Tuple consisting of the list of activations and units for each
            layer

        """
        if wb is None:
            weights_list, biases_list = self.weights_list, self.biases_list
        else:
            weights_list, biases_list = wb

        activations = [x]
        units = [x]
        z = x
        for i in range(self.nb_layers):
            # Note that the biases_list dimension is runtime resized so as to
            # compute the desired operation
            a = z.dot(weights_list[i]) + biases_list[i]
            activations.append(a)
            z = self.activation_functions[i](a)
            units.append(z)

        self.y = z

        return activations, units

    # %% backpropagation

    def get_gradients(self, x, t, beta=None, wb=None):
        """Backpropagation algorithm computing the gradient for both the
           weights and biases on each layer


        Parameters
        ----------
        x : np.array
            (N,D0) matrix holding each input data sample.
            The i-th data sample is stored in the i-th row
        t : np.array
            (N,Dr) matrix representing labels for each data sample. If these
            labels correspond to a binary classification problems (Dr = 1),
            there are as many labels as input data samples. On the other hand,
            if we have a multiclass classification problem (Dr > 1),
            labels for each sample are row-wise stacked
        beta : int
            Regularization parameter. If the parameter is not passed, then we
            use the MLP attribute which has been previously set
        wb : tuple
            Tuple consisting of a weights and a biases list to be used for
            computing the feed forward result. If this parameter is not passed
            then the MLP own weights are used for the computations

        Notes
        ----------
        Slightly different from the class notes due to the separation of bs
        and Ws and the change made to index the weights.

        delta_k matrixes have shape (N,Dk)


        Returns
        -------
        Tuple
            Tuple consisting of the list of gradients and biases for each layer
            Note that k-th index is (k+1)-th layer gradient since layer 0
            (input) has no Ws

        """

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

            # Computing new delta values
            if k < self.nb_layers:
                # Weights of the (k+1)-th layer
                w = weights_list[k]
                # Obtain derivative activation function on layer k
                dh = self.diff_activation_functions[k - 1]
                # activations from layer k
                a = activations[k]
                delta_k = (delta_k1.dot(w.T)) * dh(a)
            else:
                # We can assume the derivative of En respect to the last
                # activations layer is y-t
                delta_k = units[k] - t

            if beta is None:
                b = self.beta
            else:
                b = beta

            # Adding the regularization term for the Ws gradient
            reg_term = None
            if self.reg_method is None:
                reg_term = 0
            elif self.reg_method == 'L1':
                reg_term = (b * np.sign(weights_list[k - 1]))
            elif self.reg_method == 'L2':
                reg_term = (b * weights_list[k - 1])
            #  Elastic net regularization
            elif self.reg_method == 'Elastic_Net':
                reg_term = (b * (np.sign(weights_list[k - 1]) +
                                 weights_list[k - 1]))

            # Thanks to the einsum function we can avoid using another for loop
            # See that gradients are averaged because they are computed at the
            # for all the input data
            grad_wk = (np.einsum(
                'ij,ik', units[k - 1], delta_k) / N) + reg_term
            grad_w_list[k - 1] = grad_wk

            grad_bk = np.sum(delta_k, axis=0) / N
            grad_b_list[k - 1] = grad_bk

            delta_k1 = delta_k

        return grad_w_list, grad_b_list

    # %% training method for the MLP

    def train(self, x_data, t_data, epochs, batch_size,
              initialize_weights=False, print_cost=False, beta=0,
              reg_method=None, **opt_args):
        """Trains the Multilayer Perceptron using certain hyperparameters


        Parameters
        ----------
        x_data : np.array
            (N,D0) matrix holding each input data sample. The i-th data sample
            is stored in the i-th row.
        t_data : np.array
            (N,Dr) matrix representing labels for each data sample. If these
            labels correspond to a binary classification problems (Dr = 1),
            there are as many labels as input data samples. On the other hand,
            if we have a multiclass classification problem (Dr > 1),
            labels for each sample are row-wise stacked
        epochs : int
            Number of epochs to be used to train the model
        batch_size : int
            Number of data samples to be considered in an epoch
        initialize_weights : bool
            Boolean flag which decides if a weight initialization has to be
            performed
        print_cost : bool
            Boolean flag which can be used to display the current cost during
            the train proccess
        beta : int
            Regularization parameter. Its default value is 0 (ie, no regulari-
            zation)
        reg_method : string
            Indicates the regularization method to be used. There have been
            implemented: 'L1', 'L2' and 'Elastic_Net' regularizations


        Returns
        -------
        None

        """
        self.beta = beta
        self.reg_method = reg_method
        opt = mlpo.Optimizer.get_optimizer(self, **opt_args)

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
              epochs=1000, batch_size=20, initialize_weights=False,
              method='adam',
              eta=0.1,
              beta=0,
              gamma=0.9,
              beta_1=0.9,
              beta_2=0.999,
              epsilon=1e-8,
              print_cost=True)
