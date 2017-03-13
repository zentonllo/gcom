# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt

__author__ = "Ignacio Casso, Daniel Gamo, Gwydion J. Martín, Alberto Terceño"


# Sigmoid function used as the activation function
def sigmoid(z):
    return 1/(1+np.exp(-z))


class Perceptron(object):

    # Setting empty weight matrix
    def __init__(self):
        self.bW = np.array([])
        pass

    # x:numpy.array(N,D+1) (First column of x is already filled up with 1's)
    def get_nn_value(self, x):
        # Returns an array with N components (or number if N=1)
        return sigmoid(x.dot(self.bW))

    # x:numpy.array(N,D) (First column of x is NOT filled up with 1's)
    # t: numpy.array(N)
    def train(self, x, t, epochs=20, batch_size=5, epsilon=0.1):
        n = x.shape[0]
        D = x.shape[1]
        self.bW = np.zeros(D+1)
        onesColumn = np.array([[1]*n]).T
        x = np.hstack((onesColumn, x))

        ind = range(n)
        for _ in range(epochs):
            np.random.shuffle(ind)
            # When (n 'mod' batch_size) is not 0 we discard the spare ones
            for i in range(0, n+1-batch_size, batch_size):
                    indexes = ind[i:i+batch_size]
                    grad_bW = self.get_grad(x[indexes], t[indexes])
                    self.bW = self.bW - epsilon*grad_bW

    # x:numpy.array(N,D) (First column of x is NOT filled up with 1's)
    def classify(self, x):
        n = x.shape[0]
        onesColumn = np.array([[1]*n]).T
        x = np.hstack((onesColumn, x))
        outputs = self.get_nn_value(x)
        for i in range(n):
            if (outputs[i] <= 0.5):
                outputs[i] = 0
            else:
                outputs[i] = 1

        return outputs

    # x: numpy.array(N,D+1) (First column of x is already filled up with 1's);
    # t: numpy.array(N)
    def get_grad(self, x, t):
        # We compute derivatives respect to Wi's for each data sample,
        # that is: (Y-T)*Xi
        n = x.shape[0]

        # Array (y1-t1, ... , yn-tn)
        delta = self.get_nn_value(x)-t
        # Gradient matrix (NxD+1)
        GradMatrix = delta[:, np.newaxis] * x
        # Sum matrix rows and returns a D+1 array
        grad_bW = np.sum(GradMatrix, axis=0) / n
        return grad_bW

    # When D = 2, this method plots the points generated and the line
    # which separates these two classes after the train has been made
    # x:numpy.array(N,D) (First column of x is NOT filled up with 1's)
    def plot(self, x):
        D = x.shape[1]
        if (D == 2):
            x_min, x_max = np.min(x)-5, np.max(x)+5
            abscissas = np.array([x_min, x_max])
            bW = self.bW
            if (bW[2] != 0):
                ordinates = -bW[1] / bW[2] * abscissas - bW[0] / bW[2]
            else:
                aux = -bW[0] / bW[1]
                abscissas = np.array([aux, aux])
                ordinates = np.array([-19, 19])
            plt.axis('equal')
            plt.scatter(x[:, 0], x[:, 1])
            plt.plot(abscissas, ordinates)
            plt.show()
        else:
            print ("plot(x) can only be called when D = 2")
    pass

if __name__ == '__main__':
    # Test made for 200 points
    nb_black = 100
    nb_red = 100
    x_data_black = np.random.randn(nb_black, 2) + np.array([0, 0])
    x_data_red = np.random.randn(nb_red, 2) + np.array([10, 10])

    x_data = np.vstack((x_data_black, x_data_red))
    t = np.asarray([0]*nb_black + [1]*nb_red)

    perceptron = Perceptron()
    perceptron.train(x_data, t, epochs=1000, batch_size=40, epsilon=0.1)
    perceptron.plot(x_data)
