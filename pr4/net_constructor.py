# -*- coding: utf-8 -*-
"""
Module modeling a Neural Network, containing methods to create several types
of layers using the TensorFlow API.

"""
from __future__ import print_function, division

import tensorflow as tf
import numpy as np
import sys
from datetime import datetime
from tensorflow.examples.tutorials.mnist import input_data

NOW = datetime.utcnow().strftime("%Y%m%d%H%M%S")
ROOT_LOGDIR = 'tf_logs'
LOG_DIR = "{}/run-{}".format(ROOT_LOGDIR, NOW)

__author__ = "Ignacio Casso, Daniel Gamo, Gwydion J. Martín, Alberto Terceño"


class NetConstructor(object):
    """Class that models a Neural Network using the TensorFlow API

    Attributes
    ----------
    file_writer :
        FileWriter class used for saving an event file
    train_step :
        Operation that updates the loss for the current graph
    layers :
        List of layers used in order to create the network
    layers_dict :
        Dictionary containing the available layers for our network
    activation_dict :
        Dictionary containing the available activation functions for our
        network
    loss_dict :
        Dictionary containing the available loss functions for our network
    """

    def __init__(self, layer_list):
        """__init__ method for the NetConstructor class

        Sets up parameters for the NetConstructor class and calls the
        create_net method in order to initialize the network's structure

        Parameters
        ----------
        layers :
           List containing the layers used to create the Neural Network

        """
        tf.reset_default_graph()
        self.file_writer = None

        self.layers_dict = {'fc': self.fc_layer,
                            'conv': self.conv_layer,
                            'maxpool': self.maxpool_layer,
                            'dropout': self.dropout_layer,
                            'LRN': self.LRN_layer}

        self.activations_dict = {'relu': tf.nn.relu,
                                 'sigmoid': tf.nn.sigmoid,
                                 'tanh': tf.nn.tanh,
                                 'identity': tf.identity,
                                 'softmax': tf.nn.softmax}
        # By default, applies these functions with the right dimensions

        self.loss_dict = {'softmax': tf.nn.softmax_cross_entropy_with_logits,
                          #'identity': tf.nn.l2_loss,
                          'sigmoid': tf.nn.sigmoid_cross_entropy_with_logits}

        self.create_net(layer_list)

    def conv_layer(self, inputs, layer_info):
        """Method that creates a TensorFlow's Convolutional Layer

        Parameters
        ----------
        inputs :
            Placeholder for the tensor representing input data samples
        layer_info :
            Parameters used to build the Convolutional Layer

        Returns
        -------
        layer
            TensorFlow's Convolutional Layer built with the layer_info
            parameters

        """
        params = {}
        params['inputs'] = inputs
        params['filters'] = layer_info['channels']
        params['kernel_size'] = layer_info['k_size']
        params['strides'] = reversed(layer_info['strides'])
        params['padding'] = layer_info['padding']
        params['activation'] = self.activations_dict[layer_info['activation']]
        init_w = layer_info['init_w']
        if init_w is 'truncated_normal':
            params['kernel_initializer'] = tf.truncated_normal_initializer(stddev = layer_info['stddev_w'])
        else:
            params['kernel_initializer'] = tf.zeros_initializer()
        init_b = layer_info['init_b']
        if init_b is 'truncated_normal':
            params['bias_initializer'] = tf.truncated_normal_initializer(stddev = layer_info['stddev_b'])
        #sin else, ya es zeros por defecto
        return tf.layers.conv2d(**params)

    def maxpool_layer(self, inputs, layer_info):
        """Method that creates a TensorFlow's Pooling Layer

        Parameters
        ----------
        inputs :
            Placeholder for the tensor representing input data samples
        layer_info :
            Parameters used to build the Pooling Layer

        Returns
        -------
        layer
            TensorFlow's Pooling Layer built with the layer_info
            parameters

        """
        params = {}
        params['inputs'] = inputs
        params['pool_size'] = layer_info['k_size']
        params['strides'] = layer_info['strides']
        params['padding'] = layer_info['padding']

        return tf.layers.max_pooling2d(**params)

    def fc_layer(self, inputs, layer_info):
        """Method that creates a TensorFlow's Fully-Connected Layer

        Parameters
        ----------
        inputs :
            Placeholder for the tensor representing input data samples
        layer_info :
            Parameters used to build the Fully-Connected Layer

        Returns
        -------
        layer
            TensorFlow's Fully-Connected Layer built with the layer_info
            parameters

        """
        inputs_dim = inputs.get_shape().as_list()
        if len(inputs_dim) is 2:
            inputs_flat = inputs
        else:
            inputs_flat = tf.reshape(inputs, [-1, np.prod(inputs_dim[1:])])

        params = {}
        params['inputs'] = inputs_flat
        params['units'] = layer_info['dim']  # Int value
        params['activation'] = self.activations_dict[layer_info['activation']]
        init_w = layer_info['init_w']
        if init_w is 'truncated_normal':
            params['kernel_initializer'] = tf.truncated_normal_initializer(stddev = layer_info['stddev_w'])
        else:
            params['kernel_initializer'] = tf.zeros_initializer()
        init_b = layer_info['init_b']
        if init_b is 'truncated_normal':
            params['bias_initializer'] = tf.truncated_normal_initializer(stddev = layer_info['stddev_b'])
        #sin else, ya es zeros por defecto

        return tf.layers.dense(**params)

    def dropout_layer(self, unit, layer_info):
        """Method that creates a TensorFlow's Dropout Layer

        Parameters
        ----------
        unit :
            Placeholder for the tensor representing input data samples
        layer_info :
            Parameters used to build the Dropout Layer

        Returns
        -------
        layer
            TensorFlow's Dropout Layer built with the layer_info
            parameters

        """
        keep_prob = layer_info['prob']
        prob = tf.placeholder(tf.float32)
        self.dropouts_dic[prob] = keep_prob
        self.dropout_ones_dic[prob] = 1.

        return tf.nn.dropout(unit, prob)

    def LRN_layer(self, inputs, layer_info):
        """Method that creates a TensorFlow's Local Response Normalization
           Layer

        Parameters
        ----------
        inputs :
            Placeholder for the tensor representing input data samples
        layer_info :
            Parameters used to build the Local Response Normalization Layer

        Notes
        -----
        If there is no init_b (initialize bias) parameter in the layer info,
        no bias will be applied.

        Returns
        -------
        layer
            TensorFlow's Local Response Normalization Layer built with the
            layer_info parameters

        """
        k = layer_info['k']
        alpha = layer_info['alpha']
        beta = layer_info['beta']
        r = layer_info['r']

        return tf.nn.local_response_normalization(input=inputs,
                                                  depth_radius=r,
                                                  bias=k, alpha=alpha,
                                                  beta=beta, name='LRN_layer')

    def create_net(self, layers):
        """Method that creates the Neural Network given a list of layers

        Parameters
        ----------
        layers :
            List of layers used to build the structure of the Neural Network.
            Each layer contains a set of parameters used by the specific layer
            creation method.

        Notes
        -----
        We assume that the last layer is a FC (Fully-Connected) layer.

        Returns
        -------
        None

        """
        dim_input = layers[0]['dim']
        dim_output = (layers[-1]['dim'],)
        reshape = layers[0].get('reshape', False)

        # t_data
        self.t = tf.placeholder(tf.float32,
                                shape=(None,)+dim_output, name='t')

        if reshape:
            # x_data
            self.x = tf.placeholder(tf.float32,
                                    shape=(None,)+(np.prod(list(dim_input)),))
            Z = tf.reshape(self.x, shape=(-1,)+dim_input, name='X')
        else:
            # x_data
            self.x = tf.placeholder(tf.float32,
                                    shape=(None,)+dim_input, name='X')
            Z = self.x

        self.dropouts_dic = {}
        self.dropout_ones_dic = {}

        act = layers[-1]['activation']
        layers[-1]['activation'] = 'identity'

        for layer in layers[1:]:
            layer_type = layer.pop('type')
            Z = self.layers_dict[layer_type](Z, layer)
        self.logits = Z
        self.y = self.activations_dict[act](self.logits)

        # We assume that the last layer is FC
        with tf.name_scope('loss'):

            loss_fn = self.loss_dict[act]
            self.loss = tf.reduce_mean(loss_fn(logits=self.logits, labels=self.t), name='loss')

        self.saver = tf.train.Saver()
        self.file_writer = tf.summary.FileWriter(LOG_DIR,
                                                 tf.get_default_graph())

    @staticmethod
    def parse_optimizer(method):
        """Parses parameters used by the Network optimizer method, using the
           TensorFlow's implementation of these methods

        Parameters
        ----------
        method : tuple
            Tuple consisting of the name of the optimization method and a
            dictionary of parameters for that method.

        Notes
        -----
        Both Momentum and Nesterov use the Momentum Optimizer. If the
        parameter 'use_nesterov' is TRUE, the Nesterov Momentum method
        is used. By default, 'use_nesterov' is set to FALSE.

        Returns
        -------
        Method
            Optimization method to be used in the training

        """
        name, params = method

        dict_methods = {'SGD': tf.train.GradientDescentOptimizer,
                        'momentum': tf.train.MomentumOptimizer,
                        'nesterov': tf.train.MomentumOptimizer,
                        'adagrad': tf.train.AdagradOptimizer,
                        'adadelta': tf.train.AdadeltaOptimizer,
                        'RMSProp': tf.train.RMSPropOptimizer,
                        'adam': tf.train.AdamOptimizer}

        kwargs = {}
        kwargs['learning_rate'] = params['eta']

        if name is 'momentum':
            kwargs['momentum'] = params['gamma']
        elif name is 'nesterov':
            kwargs['momentum'] = params['gamma']
            kwargs['use_nesterov'] = True
        elif name is 'adadelta':
            kwargs['rho'] = params['gamma']
            kwargs['epsilon'] = params['epsilon']
        elif name is 'RMSprop':
            kwargs['decay'] = params['gamma']
            kwargs['epsilon'] = params['epsilon']
        elif name is 'adam':
            kwargs['beta1'] = params['beta_1']
            kwargs['beta2'] = params['beta_2']
            kwargs['epsilon'] = params['epsilon']

        return dict_methods[name](**kwargs)

    def train(self, x_train, t_train, method=('adam', {'eta': 0.001}),
              nb_epochs=1000, batch_size=10, seed='seed_nb', print_cost = True):
        """Trains the Neural Network created by the NetConstructor class.

        Parameters
        ----------
        x_train :
            Input data samples used for training.
        t_train :
            Labels for each data sample
        method : Tuple
            Optimization method to be used in the training
        nb_epochs : int
            Number of epochs to be used to train the model
        batch_size : int
            Number of data samples to be considered in an epoch
        seed :
            Seed used in order to initialize weights

        Returns
        -------
        None

        """
        # Define loss and optimizer
        opti = NetConstructor.parse_optimizer(method)
        optimizer = opti.minimize(self.loss)

        nb_data = x_train.shape[0]
        index_list = np.arange(nb_data)
        nb_batches = nb_data // batch_size

        # Some optimizers have global variables
        self.init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(self.init)
            step = 1
            for epoch in range(nb_epochs):
                np.random.shuffle(index_list)
                for batch in range(nb_batches):
                    batch_indices = index_list[batch * batch_size:
                                               (batch + 1) * batch_size]
                    x_batch = x_train[batch_indices, :]
                    t_batch = t_train[batch_indices, :]
                    feed_dict = {self.x: x_batch, self.t: t_batch}
                    feed_dict.update(self.dropouts_dic)
                    sess.run(optimizer, feed_dict=feed_dict)

                if print_cost:
                    feed_dict = {self.x: x_train, self.t: t_train}
                    feed_dict.update(self.dropout_ones_dic)
                    loss = sess.run(self.loss, feed_dict=feed_dict)
                    print("Epoch " + str(epoch) + ", Loss= " +
                          "{:.6f}".format(loss))
            self.saver.save(sess, "./MLP.ckpt")

    def predict(self, x_test):
        """Method used to predict the labels of some data samples.

        Parameters
        ----------
        x_test :
            Input data sample to be used in prediction

        Returns
        -------
        y_pred :
              Predicted labels using the Neural Network for each input data
              sample.

        """
        with tf.Session() as sess:
            self.saver.restore(sess, "./MLP.ckpt")
            feed_dict = {self.x : x_test}
            feed_dict.update(self.dropout_ones_dic)
            y_pred = sess.run(self.y, feed_dict=feed_dict)
        return y_pred
