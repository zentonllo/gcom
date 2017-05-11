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

#NOW = datetime.utcnow().strftime("%Y%m%d%H%M%S")
#ROOT_LOGDIR = 'tf_logs'
#LOG_DIR = "{}/run-{}".format(ROOT_LOGDIR, NOW)


class NetConstructor(object):
    def __init__(self, layers):
        tf.reset_default_graph()
        self.file_writer = None
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
                                 'identity': tf.identity,
				 'softmax': tf.nn.softmax} #por defecto lo hace en la dimension correcta, creo
        self.loss_dict = {'softmax': tf.nn.softmax_cross_entropy_with_logits,
                          'identity': tf.nn.l2_loss,
                          'sigmoid': tf.nn.sigmoid_cross_entropy_with_logits}
                             
        
        self.create_net()

    
    
    # Fully-connected layer
    def fc_layer(self, inputs, layer_info):
        
        # Hacer reshape 
        inputs_dim = inputs.get_shape().as_list()
        if len(inputs_dim) == 4:
            prod = inputs_dim[1]*inputs_dim[2]*inputs_dim[3]
            inputs = tf.reshape(inputs, [-1, prod])
        
        dim1 = inputs.get_shape().as_list()[1]
        dim2 = layer_info['dim']
        
        activation_fn = layer_info['activation']

        with tf.name_scope('fc_layer'):
            w_shape = [dim1, dim2]
            # Habría que usar el 'init' del diccionario?
            w = tf.Variable(tf.truncated_normal(w_shape), name='weights')
            b = tf.Variable(tf.zeros(dim2), name='bias')
            p = tf.matmul(inputs, w)
            a = tf.add(p, b, name='activation')
            h = self.activations_dict[activation_fn]
            z = h(a, name='unit')
            return z
            
    def fc_layer2(self, inputs, layer_info):

        return tf.layers.conv2d(inputs, layer_info['dim'], inputs.get_shape()[1:3] ) # weights_initializer, biases_initializer
    
    #Dropout
    def dropout_layer(self, unit, layer_info):
        return tf.layers.dropout(unit, layer_info['prob']) #prob es la probabilidad de quitarlos, tal vez queramos usar 1-prob 
        
    #Conv
    def conv_layer(self, unit, layer_info):
        
        
        hor_stride, ver_stride = layer_info['stride']
        k1, k2 = layer_info['kernel_size']
        padding = layer_info['padding']
        in_channels = int(unit.get_shape()[3])
        out_channels = layer_info['channels']
        activation_fn = layer_info['activation']

        with tf.name_scope('conv_layer'):
            
            weights = tf.Variable(tf.random_normal([k1, k2, in_channels, out_channels]))
            biases = tf.Variable(tf.zeros([out_channels]))
            x = tf.nn.conv2d(unit, weights, strides=[1, ver_stride, hor_stride, 1], padding = padding)
            x = tf.nn.bias_add(x, biases)
            return self.activations_dict[activation_fn](x)
    
    def conv_layer2(self, inputs, layer_info):
        
        layer_info['inputs']=inputs
        layer_info['filters'] = layer_info.pop('channels')
        layer_info['kernel_size'] = layer_info.pop('k_size')
        layer_info['strides'] = reversed(layer_info['strides'])
        #padding y activation se llaman igual, faltan kernel_initializer y bias initializer

        return tf.layers.conv2d(**layer_info)

    #Maxpool
    # el stride y el kernel_size deberían ser iguales?  Respuesta: si acepta padding no necesariamente
    def maxpool_layer(self, inputs, layer_info):
        return tf.layers.max_pooling2d(inputs, layer_info['ksize'], layer_info['strides'], padding = layer_info['padding'])
    
    
    #LRN
    def LRN_layer(self, inputs, layer_info):
        """
        k = layer_info['k']
        alpha = layer_info['alpha']
        beta = layer_info['beta']
        r = layer_info['r']
        return tf.nn.local_response_normalization(input=unit, depth_radius=r, bias=k, alpha=alpha, beta=beta, name='LRN_layer')
        """
        return inputs
    
    def create_net(self):
        
        #En las especificaciones pone que la dimension inicial debe ser una tupla, aunque luego pone
		#(784) en vez de (784,) como ejemplo de dimension 1. Nosotros asumiremos que es una tupla
        dim_input = self.layers[0]['dim']
        
        dim_output = (self.layers[-1]['dim'],) #asumimos que la salida tiene dimension 1 (es decir (N,1))
        
        self.X = tf.placeholder(tf.float32, shape=(None,)+dim_input, name='X') #x_data
        self.t = tf.placeholder(tf.float32, shape=(None,)+dim_output, name='t')#t_data

        Z = self.X
        for layer in self.layers[1:]:
            layer_type = layer.pop('type')
            Z  = self.layers_dict[layer_type](Z, layer)
        self.y = Z
        
        with tf.name_scope('loss'):	#Suponemos por ahora que la última capa es fc
            loss_fn = self.loss_dict[self.layers[-1]['activation']]
            self.loss = tf.reduce_mean(loss_fn(logits=self.y, labels=self.t), name='loss')

        self.init = tf.global_variables_initializer()

        self.saver = tf.train.Saver()
#        self.file_writer = tf.summary.FileWriter(LOG_DIR, tf.get_default_graph())    
                
    @staticmethod
    def parsea_optimizador(method):

        name, params = method

        dict_methods = {'SGD': tf.train.GradientDescentOptimizer,
                        'momentum' : tf.train.MomentumOptimizer,
                        'nesterov' : tf.train.MomentumOptimizer,
                        'adagrad' : tf.train.AdagradOptimizer,
                        'adadelta' : tf.train.AdadeltaOptimizer,
                        'RMSProp' : tf.train.RMSPropOptimizer,
                        'adam' : tf.train.AdamOptimizer}


        params['learning_rate'] = params.pop('eta') #comun a todos

        if name is 'momentum':
            params['momentum'] = params.pop('gamma')
        elif name is 'nesterov':
            params['momentum'] = params.pop('gamma')
            params['use_nesterov'] = True
        elif name is 'adagrad':
            if 'epsilon' in params:
                del params['epsilon'] #el adagrad de tensorflow no acepta epsilon
        elif name is 'adadelta':
            params['rho'] = params.pop('gamma')
        elif name is 'RMSprop':
            params['decay'] = params.pop('gamma')
        #beta1, beta2 y epsilon se llaman igual


        return dict_methods[name](**params)

       
    def train(self, x_train, t_train,
              nb_epochs=1000,
              batch_size=10,
              method=('adam', {'eta':0.001, 'beta1':0.9, 'beta2':0.999, 'epsilon':1e-8}),
              seed=3):
        
        optimizer = self.parsea_optimizador(method)
        self.train_step = optimizer.minimize(self.loss, name='train_step')

        nb_data = x_train.shape[0]
        index_list = np.arange(nb_data)
        nb_batches = nb_data // batch_size #que diferencia hay entre / y //?

        self.init = tf.global_variables_initializer() #algunos optimizadores tienen variables globales, como adam
        self.sess = tf.Session()
        self.sess.run(self.init)
        for epoch in range(nb_epochs):
            np.random.shuffle(index_list)
            for batch in range(nb_batches):
                batch_indices = index_list[batch * batch_size:
                                           (batch + 1) * batch_size]
                x_batch = x_train[batch_indices, :]
                t_batch = t_train[batch_indices, :]
                self.sess.run(self.train_step,
                         feed_dict={self.X: x_batch,
                                    self.t: t_batch})
            cost = self.sess.run(self.loss, feed_dict={self.X: x_train,
                                                  self.t: t_train})
            sys.stdout.write('cost=%f %d\r' % (cost, epoch))
            sys.stdout.flush()
#            self.saver.save(sess, LOG_DIR)

    def predict(self, x_test):
        with tf.Session() as sess:
            #sess.run(self.init)
#            self.saver.restore(sess, LOG_DIR)
            y_pred = self.sess.run(self.y, feed_dict={self.X: x_test})
        return y_pred


"""



No esta claro en strides y ksize que componente es la vertical y cual es la horizontal.
Como normalmente son cuadradas no importa demasiado, pero puede que lo tengamos al reves












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
