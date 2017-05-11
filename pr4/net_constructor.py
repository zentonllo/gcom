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

    
             
    def fc_layer(self, inputs, layer_info):


        inputs_dim = inputs.get_shape().as_list()
        if len(inputs_dim) is 2:
            inputs_flat = inputs
        else:
            inputs_flat = tf.reshape(inputs, [-1, np.prod(inputs_dim[1:])])


        params = {}
        params['inputs'] = inputs_flat
        params['units'] = layer_info['dim']#es un entero, segun las especificaciones
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

    #Dropout
    def dropout_layer(self, unit, layer_info):
        return tf.layers.dropout(unit, layer_info['prob']) #prob es la probabilidad de quitarlos, tal vez queramos usar 1-prob 
        
    
    def conv_layer(self, inputs, layer_info):
        
        params = {}
        params['inputs']=inputs
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

    #Maxpool
    # el stride y el kernel_size deberían ser iguales?  Respuesta: si acepta padding no necesariamente
    def maxpool_layer(self, inputs, layer_info):
        return tf.layers.max_pooling2d(inputs, layer_info['k_size'], layer_info['strides'], padding = layer_info['padding'])
    
    
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
        self.file_writer = tf.summary.FileWriter(LOG_DIR, tf.get_default_graph())    
                
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
                                    self.t: t_batch})
                cost = sess.run(self.loss, feed_dict={self.X: x_train,
                                                  self.t: t_train})
                sys.stdout.write('cost=%f %d\r' % (cost, epoch))
                sys.stdout.flush()
            self.saver.save(sess, "./MLP.ckpt")

    def predict(self, x_test):
        with tf.Session() as sess:
            self.saver.restore(sess, "./MLP.ckpt")
            y_pred = sess.run(self.y, feed_dict={self.X: x_test})
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
