# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 10:40:14 2017

@author: alumno
"""

from __future__ import print_function, division

import tensorflow as tf
import numpy as np
import sys
from datetime import datetime

NOW = datetime.utcnow().strftime("%Y%m%d%H%M%S")
ROOT_LOGDIR = 'tf_logs'
LOG_DIR = "{}/run-{}".format(ROOT_LOGDIR, NOW)


class NetConstructor(object):
    def __init__(self, layers):
        tf.reset_default_graph()
        self.file_writer = None
        self.logits = None
        self.output = None
        self.train_step = None
        self.layers  = layers
        self.layers_dict = { 'fc': self.fc_layer,
                             'conv': self.conv_layer,
                             'maxpool': self.maxpool_layer,
                             'dropout': self.dropout_layer,
                             'LRN': self.LRN_layer}
                             
        self.activations_dict = {'relu': tf.nn.relu,
                                 'sigmoid': tf.nn.sigmoid,
                                 'tanh': tf.nn.tanh,
                                 'identity': tf.identity}
        self.loss_dict = {'softmax': tf.nn.softmax_cross_entropy_with_logits,
                          'identity': tf.nn.l2_loss,
                          'sigmoid': tf.nn.sigmoid_cross_entropy_with_logits}
                             
        
        self.create_net()

    
    
    # Fully-connected layer
    def fc_layer(self, unit, layer_info):
        
        dim = layer_info['dim']
        
        activation_fn = layer_info['activation']

        with tf.name_scope('fc_layer'):
            w_shape = (int(unit.get_shape()[1]), dim)
            # Habría que usar el 'init' del diccionario?
            w = tf.Variable(tf.truncated_normal(w_shape), name='weights')
            b = tf.Variable(tf.zeros(dim), name='bias')
            a = tf.add(tf.matmul(unit, w), b, name='activation') #arregladlo
            z = None
            if activation_fn in self.activations_dict:
                h = self.activations_dict[activation_fn]
                z = h(a, name='unit')
            else:
               	h = self.loss_dict[activation_fn]
               	self.logits = a
               	z = h(logits=z, labels=self.y)
               	self.output = z
            return z
            
    
    #Dropout
    def dropout_layer(self, unit, layer_info):
        keep_prob = layer_info['prob']
        return tf.nn.dropout(unit, keep_prob)   
        
        
    #Conv
    def conv_layer(self, unit, layer_info):
        hor_stride, ver_stride = layer_info['stride']
        k1, k2 = layer_info['kernel_size']
        padding = layer_info['padding']
        in_channels = int(unit.get_shape()[3])
        out_channels = layer_info['channels']
        return tf.nn.conv2d(input=unit, filter=[k1,k2,in_channels,out_channels], 
                            strides=[1,ver_stride,hor_stride, 1], 
                            padding=padding, name='conv_layer') 
    
    #Maxpool
    # el stride y el kernel_size deberían ser iguales?
    def maxpool_layer(self, unit, layer_info):
        ver_stride, hor_stride = layer_info['stride']
        k1, k2 = layer_info['ksize']
        padding = layer_info['padding']
        return tf.nn.max_pool(value=unit, ksize=[1,k1,k2,1], strides=[1,ver_stride,hor_stride, 1], padding=padding)    
    
    
    #LRN
    def LRN_layer(self, unit, layer_info):
        k, alpha, beta, r = layer_info['LRN_params'] 
        return tf.nn.local_response_normalization(input=unit, depth_radius=r, bias=k, alpha=alpha, beta=beta, name='LRN_layer')
    
    def create_net(self):
        
        #  Parseo de dimensiones (cuando es convolucional se pone tupla, cuando es fc se pone número sin tupla)
        nb_input = self.layers[0]['dim']
        if type(nb_input) is not tuple:
            nb_input = (nb_input,)
        
        nb_ouput = self.layers[-1]['dim']
        
        self.X = tf.placeholder(tf.float32, shape=(None,)+nb_input, name='X')
        self.y = tf.placeholder(tf.float32, shape=(None, nb_ouput), name='y')
        nb_layers = len(self.layers)
        Z = self.X
        for layer in range(1, nb_layers):
            layer_type = self.layers[layer]['type']
            Z  = self.layers_dict[layer_type](Z, self.layers[layer])
        
        self.loss = tf.reduce_mean(self.output,name='loss')

        self.init = tf.global_variables_initializer()

        self.saver = tf.train.Saver()
        self.file_writer = tf.summary.FileWriter(LOG_DIR, tf.get_default_graph())    
                
    @staticmethod
    def parsea_optimizador(method):
        name, params = method
        if name is 'adam':
            return tf.train.AdamOptimizer(params['lr'], params['beta1'], params['beta2'], params['epsilon'])
        elif name is 'adadelta':
            return tf.train.AdadeltaOptimizer(params['lr'], params['rho'], params['epsilon'])
        elif name is 'adagrad':
            return tf.train.AdagradOptimizer(params['lr'], params['initial_accumulator'])
        elif name is 'gradient_descent':
            return tf.train.GradientDescentOptimizer(params['lr'])
        elif name is 'momentum':
            return tf.train.MomentumOptimizer(params['lr'], params['momentum_value'])
        elif name is 'proximal_adagrad':
            return tf.train.ProximalAdagradOptimizer(params['lr']) 
        elif name is 'proximal_gradient_descent':
            return tf.train.ProximalGradientDescentOptimizer(params['lr']) 
        elif name is 'RMSProp':
            return tf.train.RMSPropOptimizer(params['lr'], params['decay'], params['momentum'], params['epsilon'])
            
       
    def train(self, x_train, t_train,
              nb_epochs=1000,
              batch_size=10,
              method=('adam', {'lr':0.001, 'beta1':0.9, 'beta2':0.999, 'epsilon':1e-8}),
              seed=3):
        
        optimizer = parsea_optimizador(method)
        self.train_step = optimizer.minimize(self.loss, name='train_step')

        nb_data = x_train.shape[0]
        index_list = np.arange(nb_data)
        nb_batches = nb_data // batch_size

        with tf.Session() as sess:
            sess.run(self.init)
            for epoch in range(nb_epochs):
                np.random.shuffle(index_list)
                for batch in range(nb_batches):
                    batch_indices = index_list[batch * batch_size:
                                               (batch + 1) * batch_size]
                    x_batch = x_train[batch_indices, :]
                    t_batch = t_train[batch_indices, :]
                    sess.run(self.train_step,
                             feed_dict={self.X: x_batch,
                                        self.y: t_batch})
                cost = sess.run(self.loss, feed_dict={self.X: x_train,
                                                      self.y: t_train})
                sys.stdout.write('cost=%f %d\r' % (cost, epoch))
                sys.stdout.flush()
            self.saver.save(sess, LOG_DIR)

    def predict(self, x_test):
        with tf.Session() as sess:
            self.saver.restore(sess, LOG_DIR)
            pred = tf.nn.softmax(self.logits)
            y_pred = sess.run(pred, feed_dict={self.x: x_test})
        return y_pred


"""

        



layer_list es una lista de diccionarios que describirán las sucesivas capas de la red.

layer_list = [layer_0, layer_2, layer_3, ..., layer_m]

layer_0 contendrá solamente la dimensión de los datos de entrada, que será una tupla o un número:

layer_0 = {'dim': (dim_0, dim_1, ..., dim_L)}

Por ejemplo,

layer_0 = {'dim': (224, 224, 3)}

en el caso de que los datos de entrada sean imágenes de dimensión 224 x 224 y 3 canales de color o

layer_0 = {'dim': 784} en el caso de que sean vectores que representen imágenes de MNIST.

En las restantes capas, la estructura será la siguiente:

(se indican los parámetros mínimos que deberán estar implementados, se
 pueden añadir otros si se desea. No todos los parámetros deben
 aparecer siempre, por ejemplo, una capa de dropout sólo necesita la
 probabilidad de hacer dropout)

layer_k = {'type': layer_type, # tipo de capa: 'fc', 'conv', 'maxpool', 'dropout', 'LRN', ...
           'dim': (dim_0, ..., dim_L) # dimensiones de la salida
                                      # de la capa (en su caso)
           'kernel_size': size # por ejemplo, (3, 3) en una máscara convolucional 3 x 3  // kerner_size - conv    ksize - maxpool
           'stride': stride # por ejemplo, (1, 1) si se hace stride 1 horizontal y 1 vertical
           'init': init_method # método de inicialización de pesos y biases, por ejemplo
                               # ('truncated_normal', stddev, 'zeros'), 'xavier' o 'he'
           'padding': padding # 'SAME', 'VALID'
           'activation': activation, # función de activación, 
                                     # 'sigmoid', 'tanh', 'relu', 'identity', ...
           'prob': probability, # float, probabilidad usada en dropout
           'LRN_params': (k, alpha, beta, r)}
           
           
         AÑADIMOS CAMPO CHANNELS


El método train entrenará la red, recibiendo los datos de entrenamiento,
número de epochs, tamaño del batch, una semilla opcional para el
generador de números aleatorios y el método de entrenamiento:

method = (str, params),

el primer elemento describe el método de optimización,
por ejemplo, 'SGD', 'nesterov', 'momentum', 'adagrad', 'adadelta', 'RMSprop'.
El segundo elemento es un diccionario de parámetros adaptados al método,
siguiendo la notación de la práctica anterior. Por ejemplo,
method = ('SGD', {'eta': 0.1}) describirá un descenso de gradiente estocástico
con learning rate = 0.1

El método predict recibirá datos de test y devolverá la predicción de la red, una vez entrenada.


Se acompañará a la práctica dos scripts en Python llamados fc_CIFAR10.py y conv_CIFAR10.py
que usarán la clase NetConstructor para clasificar la base de datos CIFAR10. El primero sólo podrá
usar capas completamente conectadas, mientras que el segundo usará
todas las herramientas disponibles.

Módulos que se pueden importar: numpy, tensorflow


"""