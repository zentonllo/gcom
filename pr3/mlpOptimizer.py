# -*- coding: utf-8 -*-
"""Module modeling classes for various optimization techniques and
   its creation.

    The optimizations techniques implemented are the following:
        - Stochastic Gradient Descent (SGD)
        - Momentum
        - Nesterov accelerated gradient
        - Adagrad
        - Adadelta
        - RMSprop
        - Adam

    Each class implementing these optimization techniques has two methods:
        1) Initialization with the corresponding arguments.
        2) process_batch for training the MLP.


    More details about the implementation of this methods can be found on
    Sebastian Ruder's website:
        http://sebastianruder.com/optimizing-gradient-descent/index.html

"""
from __future__ import division, print_function

import numpy as np


__author__ = "Ignacio Casso, Daniel Gamo, Gwydion J. Martín, Alberto Terceño"


class Optimizer(object):
    """Class used for creating optimization methods' classes for training

    """
    @staticmethod
    def get_optimizer(mlp, **kwargs):
        """Creation of the optimization class used in training, along
        with its parameters.

        Parameters
        ----------
        mlp : MLP
            Multilayer Perceptron object to be trained.

        **kwargs :
            Keyword arguments. Name of the optimization method and the
            parameters to be used in the training process.

        Notes
        -----
            The default optimization method is SGD.

        Returns
        -------
            Optimizitation class to be used in the Multilayer
            Perceptron training.

        """
        dic_learning_methods = {'SGD': SGD, 'momentum': Momentum,
                                'nesterov': Nesterov, 'adagrad': Adagrad,
                                'adadelta': Adadelta, 'RMS_prop': RMSprop,
                                'adam': Adam}

        method_name = kwargs.pop("method", "SGD")
        method = dic_learning_methods[method_name]
        return method(mlp, **kwargs)


class SGD(Optimizer):
    """Class implementing Stochastic Gradient Descent (SGD) optimization

    Attributes
    ----------
    mlp : MLP
        Multilayer Perceptron object to be trained
    eta : float
        Parameter used to update weights and biases

    """

    def __init__(self, mlp, **kwargs):
        """__init__ method for the SGD class

        Sets up hyperparameters for the SGD class

        Parameters
        ----------
        mlp : MLP
            Multilayer Perceptron object to be trained
        **kwargs :
            Parameters used by the SGD method (eta)

        """
        self.mlp = mlp

        self.eta = kwargs.pop("eta", 0.1)

    def process_batch(self, x_data, t_data):
        """Batch process for the SGD class

        Parameters
        ----------
        x_data : np.array
            Matrix holding each input data sample
        t_data : np.array
            Matrix representing labels for each data sample

        Returns
        -------
        None

        """
        grad_w_list, grad_b_list = self.mlp.get_gradients(x_data, t_data)

        self.mlp.weights_list = [w - self.eta * grad_w
                                 for w, grad_w in zip(self.mlp.weights_list, grad_w_list)]
        self.mlp.biases_list = [b - self.eta * grad_b
                                for b, grad_b in zip(self.mlp.biases_list, grad_b_list)]


class Momentum(Optimizer):
    """Class implementing Momentum optimization

    Attributes
    ----------
    mlp : MLP
        Multilayer Perceptron object to be trained
    eta : float
        Parameter used to update weights and biases
    gamma : float
        Parameter used to update weights and biases
    v_w_list : np.array
        Update vector for the MLP weights
    v_b_list : np.array
        Update vector for the MLP biases

    """

    def __init__(self, mlp, **kwargs):
        """__init__ method for the Momentum class

        Sets up hyperparameters for the Momentum class

        Parameters
        ----------
        mlp : MLP
            Multilayer Perceptron object to be trained
        **kwargs :
            Parameters used by the Momentum method (eta, gamma)

        """
        self.mlp = mlp

        self.eta = kwargs.pop("eta", 0.1)
        self.gamma = kwargs.pop("gamma", 0.9)

        self.v_w_list = [np.zeros(w.shape) for w in mlp.weights_list]
        self.v_b_list = [np.zeros(b.shape) for b in mlp.biases_list]

    def process_batch(self, x_data, t_data):
        """Batch process for the Momentum class

        Parameters
        ----------
        x_data : np.array
            Matrix holding each input data sample
        t_data : np.array
            Matrix representing labels for each data sample

        Returns
        -------
        None

        """
        grad_w_list, grad_b_list = self.mlp.get_gradients(x_data, t_data)

        self.v_w_list = [self.gamma * v_w + self.eta *
                         grad_w for v_w, grad_w in zip(self.v_w_list, grad_w_list)]
        self.v_b_list = [self.gamma * v_b + self.eta *
                         grad_b for v_b, grad_b in zip(self.v_b_list, grad_b_list)]

        self.mlp.weights_list = [
            w - v_w for w, v_w in zip(self.mlp.weights_list, self.v_w_list)]
        self.mlp.biases_list = [b - v_b for b,
                                v_b in zip(self.mlp.biases_list, self.v_b_list)]


class Nesterov(Optimizer):
    """Class implementing Nesterov accelerated gradient optimization

    Attributes
    ----------
    mlp : MLP
        Multilayer Perceptron object to be trained
    eta : float
        Parameter used to update weights and biases
    gamma : float
        Parameter used to update weights and biases
    v_w_list : np.array
        Update vector for the MLP weights
    v_b_list : np.array
        Update vector for the MLP biases

    """

    def __init__(self, mlp, **kwargs):
        """__init__ method for the Nesterov class

        Sets up hyperparameters for the Nesterov class

        Parameters
        ----------
        mlp : MLP
            Multilayer Perceptron object to be trained
        **kwargs :
            Parameters used by the Nesterov method (eta, gamma)

        """
        self.mlp = mlp

        self.eta = kwargs.pop("eta", 0.1)
        self.gamma = kwargs.pop("gamma", 0.9)

        self.v_w_list = [np.zeros(w.shape) for w in mlp.weights_list]
        self.v_b_list = [np.zeros(b.shape) for b in mlp.biases_list]

    def process_batch(self, x_data, t_data):
        """Batch process for the Nesterov class

        Parameters
        ----------
        x_data : np.array
            Matrix holding each input data sample
        t_data : np.array
            Matrix representing labels for each data sample

        Returns
        -------
        None

        """
        future_weights_list = [w - self.gamma * v_w
                               for w, v_w in zip(self.mlp.weights_list, self.v_w_list)]

        future_biases_list = [b - self.gamma * v_b
                              for b, v_b in zip(self.mlp.biases_list, self.v_b_list)]

        grad_w_list, grad_b_list = self.mlp.get_gradients(
            x_data, t_data, wb=(future_weights_list, future_biases_list))

        self.v_w_list = [self.gamma * v_w + self.eta *
                         grad_w for v_w, grad_w in zip(self.v_w_list, grad_w_list)]
        self.v_b_list = [self.gamma * v_b + self.eta *
                         grad_b for v_b, grad_b in zip(self.v_b_list, grad_b_list)]

        self.mlp.weights_list = [
            w - v_w for w, v_w in zip(self.mlp.weights_list, self.v_w_list)]
        self.mlp.biases_list = [b - v_b for b,
                                v_b in zip(self.mlp.biases_list, self.v_b_list)]


class Adagrad(Optimizer):
    """Class implementing Adagrad optimization

    Attributes
    ----------
    mlp : MLP
        Multilayer Perceptron object to be trained
    eta : float
        Parameter used to update weights and biases
    epsilon : float
        Parameter used to update weights and biases
    G_w_list : np.array
        G_t terms for the MLP weights
    G_b_list : np.array
        G_t terms for the MLP biases

    """

    def __init__(self, mlp, **kwargs):
        """__init__ method for the Adagrad class

        Sets up hyperparameters for the Adagrad class

        Parameters
        ----------
        mlp : MLP
            Multilayer Perceptron object to be trained
        **kwargs :
            Parameters used by the Adagrad method (eta, epsilon)

        """
        self.mlp = mlp

        self.eta = kwargs.pop("eta", 0.1)
        self.epsilon = kwargs.pop("epsilon", 1e-8)

        self.G_w_list = [np.zeros(w.shape) for w in mlp.weights_list]
        self.G_b_list = [np.zeros(b.shape) for b in mlp.biases_list]

    def process_batch(self, x_data, t_data):
        """Batch process for the Adagrad class

        Parameters
        ----------
        x_data : np.array
            Matrix holding each input data sample
        t_data : np.array
            Matrix representing labels for each data sample

        Returns
        -------
        None

        """
        grad_w_list, grad_b_list = self.mlp.get_gradients(x_data, t_data)

        self.G_w_list = [G_w + (grad_w ** 2) for G_w,
                         grad_w in zip(self.G_w_list, grad_w_list)]
        self.G_b_list = [G_b + (grad_b ** 2) for G_b,
                         grad_b in zip(self.G_b_list, grad_b_list)]

        self.mlp.weights_list = [w - (self.eta / np.sqrt(G_w + self.epsilon)) * grad_w for w, G_w, grad_w in
                                 zip(self.mlp.weights_list, self.G_w_list, grad_w_list)]

        self.mlp.biases_list = [b - (self.eta / np.sqrt(G_b + self.epsilon)) * grad_b for b, G_b, grad_b in
                                zip(self.mlp.biases_list, self.G_b_list, grad_b_list)]


class Adadelta(Optimizer):
    """Class implementing Adadelta optimization

    Attributes
    ----------
    mlp : MLP
        Multilayer Perceptron object to be trained
    epsilon : float
        Parameter used to update weights and biases
    gamma : float
        Parameter used to update weights and biases
    avg_w_list : np.array
        Average weights list for the MLP
    avg_b_list : np.array
        Average biases list for the MLP
    avg_delta_w_list : np.array
        Adagrad update vector for the MLP weights
    avg_delta_b_list : np.array
        Adagrad update vector for the MLP biases

    """

    def __init__(self, mlp, **kwargs):
        """__init__ method for the Adadelta class

        Sets up hyperparameters for the Adadelta class

        Parameters
        ----------
        mlp : MLP
            Multilayer Perceptron object to be trained
        **kwargs :
            Parameters used by the Adadelta method (epsilon, gamma)

        """
        self.mlp = mlp

        self.epsilon = kwargs.pop("epsilon", 1e-8)
        self.gamma = kwargs.pop("gamma", 0.9)

        self.avg_w_list = [np.zeros(w.shape) for w in self.mlp.weights_list]
        self.avg_b_list = [np.zeros(b.shape) for b in self.mlp.biases_list]

        self.avg_delta_w_list = [np.zeros(w.shape)
                                 for w in self.mlp.weights_list]
        self.avg_delta_b_list = [np.zeros(b.shape)
                                 for b in self.mlp.biases_list]

    def process_batch(self, x_data, t_data):
        """Batch process for the Adadelta class

        Parameters
        ----------
        x_data : np.array
            Matrix holding each input data sample
        t_data : np.array
            Matrix representing labels for each data sample

        Returns
        -------
        None

        """
        grad_w_list, grad_b_list = self.mlp.get_gradients(x_data, t_data)

        self.avg_w_list = [self.gamma * avg_w + (1 - self.gamma) * (grad_w**2)
                           for avg_w, grad_w
                           in zip(self.avg_w_list, grad_w_list)]
        self.avg_b_list = [self.gamma * avg_b + (1 - self.gamma) * (grad_b**2)
                           for avg_b, grad_b
                           in zip(self.avg_b_list, grad_b_list)]

        delta_weights_list = [- (np.sqrt(avg_delta_w + self.epsilon) / np.sqrt(avg_w + self.epsilon)) * grad_w
                              for avg_delta_w, avg_w, grad_w in
                              zip(self.avg_delta_w_list, self.avg_w_list, grad_w_list)]

        self.mlp.weights_list = [w + delta_w
                                 for w, delta_w in
                                 zip(self.mlp.weights_list, delta_weights_list)]

        self.avg_delta_w_list = [self.gamma * avg_delta_w + (1 - self.gamma) * (delta_w**2) for avg_delta_w, delta_w in
                                 zip(self.avg_delta_w_list, delta_weights_list)]

        delta_biases_list = [- (np.sqrt(avg_delta_b + self.epsilon) / np.sqrt(avg_b + self.epsilon)) * grad_b
                             for avg_delta_b, avg_b, grad_b in
                             zip(self.avg_delta_b_list, self.avg_b_list, grad_b_list)]

        self.mlp.biases_list = [b + delta_b
                                for b, delta_b in
                                zip(self.mlp.biases_list, delta_biases_list)]

        self.avg_delta_b_list = [self.gamma * avg_delta_b + (1 - self.gamma) * (delta_b**2) for avg_delta_b, delta_b in
                                 zip(self.avg_delta_b_list, delta_biases_list)]


class RMSprop(Optimizer):
    """Class implementing RMSprop optimization

    Attributes
    ----------
    mlp : MLP
        Multilayer Perceptron object to be trained
    eta : float
        Parameter used to update weights and biases
    epsilon : float
        Parameters used to update weights and biases
    gamma : float
        Parameter used to update weights and biases
    avg_w_list : np.array
        Update vector for the MLP weights
    avg_b_list : np.array
        Update vector for the MLP biases

    """

    def __init__(self, mlp, **kwargs):
        """__init__ method for the RMSprop class

        Sets up hyperparameters for the RMSprop class

        Parameters
        ----------
        mlp : MLP
            Multilayer Perceptron object to be trained
        **kwargs :
            Parameters used by the RMSprop method (eta, epsilon, gamma)
        """
        self.mlp = mlp

        self.eta = kwargs.pop("eta", 0.001)
        self.epsilon = kwargs.pop("epsilon", 1e-8)
        self.gamma = kwargs.pop("gamma", 0.9)

        self.avg_w_list = [np.zeros(w.shape) for w in mlp.weights_list]
        self.avg_b_list = [np.zeros(b.shape) for b in mlp.biases_list]

    def process_batch(self, x_data, t_data):
        """Batch process for the RMSprop class

        Parameters
        ----------
        x_data : np.array
            Matrix holding each input data sample
        t_data : np.array
            Matrix representing labels for each data sample

        Returns
        -------
        None

        """
        grad_w_list, grad_b_list = self.mlp.get_gradients(x_data, t_data)

        self.avg_w_list = [self.gamma * avg_w + (1 - self.gamma) * (grad_w**2)
                           for avg_w, grad_w
                           in zip(self.avg_w_list, grad_w_list)]
        self.avg_b_list = [self.gamma * avg_b + (1 - self.gamma) * (grad_b**2)
                           for avg_b, grad_b
                           in zip(self.avg_b_list, grad_b_list)]

        self.mlp.weights_list = [w - (self.eta / np.sqrt(avg_w + self.epsilon)) * grad_w
                                 for w, avg_w, grad_w in
                                 zip(self.mlp.weights_list, self.avg_w_list, grad_w_list)]

        self.mlp.biases_list = [b - (self.eta / np.sqrt(avg_b + self.epsilon)) * grad_b
                                for b, avg_b, grad_b in
                                zip(self.mlp.biases_list, self.avg_b_list, grad_b_list)]


class Adam(Optimizer):
    """Class implementing Adam optimization

    Attributes
    ----------
    mlp : MLP
        Multilayer Perceptron object to be trained
    eta : float
        Parameter used to update weights and biases
    epsilon : float
        Parameter used to update weights and biases
    beta_1 : float
        Parameter used to estimate the mean of the gradients
    beta_2 : float
        Parameter used to estimate the uncentered variance of the gradients
    v_w_list : np.array
        Uncentered variance of the weights' gradients
    v_b_list : np.array
        Uncentered variance of the biases' gradients
    m_w_list : np.array
        Mean of the weights' gradients
    m_b_list : np.array
        Mean of the biases' gradients
    v_w_list2 : np.array
        Bias-corrected uncentered variance of the weights' gradients
    v_b_list2 : np.array
        Bias-corrected uncentered variance of the biases' gradients
    m_w_list2 : np.array
        Bias-corrected mean of the weights' gradients
    m_b_list2 : np.array
        Bias-corrected mean of the biases' gradients
    t_counter : int
        Counter incrementing in each iteration of the method

    """

    def __init__(self, mlp, **kwargs):
        """__init__ method for the Adam class

        Sets up hyperparameters for the Adam class

        Parameters
        ----------
        mlp : MLP
            Multilayer Perceptron object to be trained
        **kwargs :
            Parameters used by the Adam method (eta, epsilon, beta_1, beta_2)

        """
        self.mlp = mlp

        self.eta = kwargs.pop("eta", 0.001)
        self.epsilon = kwargs.pop("epsilon", 1e-8)
        self.beta_1 = kwargs.pop("beta_1", 0.9)
        self.beta_2 = kwargs.pop("beta_2", 0.999)

        self.v_w_list = [np.zeros(w.shape) for w in mlp.weights_list]
        self.v_b_list = [np.zeros(b.shape) for b in mlp.biases_list]
        self.m_w_list = [np.zeros(w.shape) for w in mlp.weights_list]
        self.m_b_list = [np.zeros(b.shape) for b in mlp.biases_list]

        self.v_w_list2 = [np.zeros(w.shape) for w in mlp.weights_list]
        self.v_b_list2 = [np.zeros(b.shape) for b in mlp.biases_list]
        self.m_w_list2 = [np.zeros(w.shape) for w in mlp.weights_list]
        self.m_b_list2 = [np.zeros(b.shape) for b in mlp.biases_list]

        self.t_counter = 1

    def process_batch(self, x_data, t_data):
        """Batch process for the Adam class

        Parameters
        ----------
        x_data : np.array
            Matrix holding each input data sample
        t_data : np.array
            Matrix representing labels for each data sample

        Returns
        -------
        None

        """
        grad_w_list, grad_b_list = self.mlp.get_gradients(
            x_data, t_data)

        self.v_w_list = [self.beta_2 * v_w + (1 - self.beta_2) * (grad_w**2)
                         for v_w, grad_w
                         in zip(self.v_w_list, grad_w_list)]

        self.v_w_list2 = [v_w / (1 - np.power(self.beta_2, self.t_counter))
                          for v_w in self.v_w_list]

        self.v_b_list = [self.beta_2 * v_b + (1 - self.beta_2) * (grad_b**2)
                         for v_b, grad_b
                         in zip(self.v_b_list, grad_b_list)]

        self.v_b_list2 = [v_b / (1 - np.power(self.beta_2, self.t_counter))
                          for v_b in self.v_b_list]

        self.m_w_list = [self.beta_1 * m_w + (1 - self.beta_1) * (grad_w)
                         for m_w, grad_w
                         in zip(self.m_w_list, grad_w_list)]

        self.m_w_list2 = [m_w / (1 - np.power(self.beta_1, self.t_counter))
                          for m_w in self.m_w_list]

        self.m_b_list = [self.beta_1 * m_b + (1 - self.beta_1) * (grad_b)
                         for m_b, grad_b
                         in zip(self.m_b_list, grad_b_list)]

        self.m_b_list2 = [m_b / (1 - np.power(self.beta_1, self.t_counter))
                          for m_b in self.m_b_list]

        self.mlp.weights_list = [w - (self.eta / (np.sqrt(v_w) + self.epsilon)) * m_w
                                 for w, v_w, m_w in
                                 zip(self.mlp.weights_list, self.v_w_list2, self.m_w_list2)]

        self.mlp.biases_list = [b - (self.eta / (np.sqrt(v_b) + self.epsilon)) * m_b
                                for b, v_b, m_b in
                                zip(self.mlp.biases_list, self.v_b_list2, self.m_b_list2)]

        self.t_counter = self.t_counter + 1
