'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function, division

import tensorflow as tf
import numpy as np
import sys
from datetime import datetime
from tensorflow.examples.tutorials.mnist import input_data

NOW = datetime.utcnow().strftime("%Y%m%d%H%M%S")
ROOT_LOGDIR = 'tf_logs'
LOG_DIR = "{}/run-{}".format(ROOT_LOGDIR, NOW)


class NetConstructor(object):

    def __init__(self, layer_list):
        tf.reset_default_graph()
        self.file_writer = None

        self.layers_dict = { 'fc': self.fc_layer,
                     'conv': self.conv_layer,
                     'maxpool': self.maxpool_layer,
                     'dropout': self.dropout_layer,
                     'LRN' : self.LRN_layer}

        self.activations_dict = {'relu': tf.nn.relu,
                                 'sigmoid': tf.nn.sigmoid,
                                 'tanh': tf.nn.tanh,
                                 'identity': tf.identity,
				 'softmax': tf.nn.softmax} #por defecto lo hace en la dimension correcta, creo

        self.loss_dict = {'softmax': tf.nn.softmax_cross_entropy_with_logits,
                          'identity': tf.nn.l2_loss,
                          'sigmoid': tf.nn.sigmoid_cross_entropy_with_logits}

        self.create_net(layer_list)

    def conv_layer(self, inputs, layer_info):
		
        params = {}
        params['inputs']=inputs
        params['filters'] = layer_info['channels']
        params['kernel_size'] = layer_info['k_size']
        params['strides'] = reversed(layer_info['strides'])
        params['padding'] = layer_info['padding']
        params['activation'] = self.activations_dict[layer_info['activation']]
        init_w = layer_info['init_w']
        if init_w is 'random_normal':
            params['kernel_initializer'] = tf.random_normal_initializer()
        init_b = layer_info['init_b']
        if init_b is 'random_normal':
            params['bias_initializer'] = tf.random_normal_initializer()
		#arreglar initializers

        return tf.layers.conv2d(**params)


	#Maxpool
    def maxpool_layer(self, inputs, layer_info):

        params = {}
        params['inputs'] = inputs
        params['pool_size'] = layer_info['k_size']
        params['strides'] = layer_info['strides']
        params['padding'] = layer_info['padding']

        return tf.layers.max_pooling2d(**params)


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
        if init_w is 'random_normal':
            params['kernel_initializer'] = tf.random_normal_initializer()
        init_b = layer_info['init_b']
        if init_b is 'random_normal':
            params['bias_initializer'] = tf.random_normal_initializer()
		#arreglar initializers

        return tf.layers.dense(**params)

	#Dropout
    def dropout_layer(self, unit, layer_info):
        keep_prob = layer_info['prob']
        prob = tf.placeholder(tf.float32)
        self.dropouts_dic[prob] = keep_prob
        self.dropout_ones_dic[prob] = 1.
	
        return tf.nn.dropout(unit, prob)

    #LRN
    def LRN_layer(self, inputs, layer_info):
        
        k = layer_info['k']
        alpha = layer_info['alpha']
        beta = layer_info['beta']
        r = layer_info['r']
        return tf.nn.local_response_normalization(input=inputs, depth_radius=r, bias=k, alpha=alpha, beta=beta, name='LRN_layer')
        

    def create_net(self, layers):


        dim_input = layers[0]['dim']
        dim_output = (layers[-1]['dim'],)
        reshape = layers[0].get('reshape', False)

        self.t = tf.placeholder(tf.float32, shape=(None,)+dim_output, name='t')#t_data

        if reshape:
            self.x = tf.placeholder(tf.float32, shape=(None,)+(np.prod(list(dim_input)),)) #x_data
            Z = tf.reshape(self.x, shape=(-1,)+dim_input, name='X')
        else:
            self.x = tf.placeholder(tf.float32, shape=(None,)+dim_input, name='X') #x_data
            Z = self.x


        self.dropouts_dic = {}
        self.dropout_ones_dic = {}

        for layer in layers[1:]:
            layer_type = layer.pop('type')
            Z  = self.layers_dict[layer_type](Z, layer)

        self.y = Z #Vease tf_mlop de valdes, a lo mejor quiere que no apliquemos la ultima activacion como el

        with tf.name_scope('loss'):	#Suponemos por ahora que la última capa es fc
			
            loss_fn = self.loss_dict[layers[-1]['activation']]
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.t), name='loss')

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



    def train(self, x_train, t_train, method=('adam', {'eta':0.001}), nb_epochs=1000, batch_size=10, seed='seed_nb', loss_name = None):

        dic_loss = {'rmsce' : tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y,
						labels=self.t))}

        display_step = 10

        # Define loss and optimizer
        if loss_name is None:
            cost = self.loss
        else:
            cost = dic_loss[loss_name]
        opti = NetConstructor.parsea_optimizador(method)
        optimizer = opti.minimize(cost)


        # Evaluate model
        correct_pred = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.t, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


        nb_data = x_train.shape[0]
        index_list = np.arange(nb_data)
        nb_batches = nb_data // batch_size

        self.init = tf.global_variables_initializer() #algunos optimizadores tienen variables globales, como adam


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
                    
                    if step % display_step == 0:
                    # Calculate batch loss and accuracy
                    	feed_dict = {self.x: x_batch, self.t: t_batch}
                    	feed_dict.update(self.dropout_ones_dic)
                    	loss, acc = sess.run([cost, accuracy], feed_dict=feed_dict)
                    	print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  			"{:.6f}".format(loss) + ", Training Accuracy= " + \
                 	 		"{:.5f}".format(acc))
                    step += 1
            self.saver.save(sess, "./MLP.ckpt")

    def predict(self, x_test):
        with tf.Session() as sess:
            self.saver.restore(sess, "./MLP.ckpt")
            y_pred = sess.run(self.y, feed_dict={self.x: x_test})
        return y_pred


if __name__ == '__main__':


    layer_list = []

    layer =  {'dim': (28, 28, 1), 'reshape':True}
    layer_list.append(layer)


    layer = {'type': 'conv',
     'channels': 32,
     'k_size': (5, 5),
     'strides': (1, 1),
     'padding': 'SAME',
     'activation': 'relu',
     'init_w': 'random_normal',
     'init_b': 'random_normal'}
    layer_list.append(layer)


    layer = {'type': 'maxpool',
     'k_size': (2, 2),
     'strides': (2, 2),
     'padding': 'SAME'}
    layer_list.append(layer)


    layer = {'type': 'conv',
     'channels': 64,
     'k_size': (5, 5),
     'strides': (1, 1),
     'padding': 'SAME',
     'activation': 'relu',
     'init_w': 'random_normal',
     'init_b': 'random_normal'}
    layer_list.append(layer)

    layer = {'type': 'maxpool',
     'k_size': (2, 2),
     'strides': (2, 2),
     'padding': 'SAME'}
    layer_list.append(layer)


    layer = {'type': 'fc',
     'dim': 1024,
     'activation': 'relu',
     'init_w': 'random_normal',
     'init_b': 'random_normal'}
    layer_list.append(layer)


    layer = {'type': 'dropout',
     'prob': 0.75}
    layer_list.append(layer)


    layer = {'type': 'fc',
     'dim': 10,
     'activation': 'identity', #loss no sabemos cual
     'init_w': 'random_normal',
     'init_b': 'random_normal'}
    layer_list.append(layer)

    red = NetConstructor(layer_list)

    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    x_train = mnist.train.images
    t_train = mnist.train.labels
    
    method = ('adam', {'eta' : 0.001, 'beta_1' : 0.9, 'beta_2' : 0.999, 'epsilon' : 1e-08})
    red.train(x_train, t_train, method = method, batch_size=128, loss_name='rmsce')
