# coding: utf-8

from __future__ import division, print_function

import pickle
import numpy as np
import download 
#from importlib import reload

import net_constructor
reload(net_constructor)
from net_constructor import NetConstructor


def maybe_download_and_extract():
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    download.maybe_download_and_extract(url, '.')

    
def process_data(file_name):

    with open(file_name, 'rb') as file:
        dict_data = pickle.load(file)
        #dict_data = pickle.load(file, encoding = 'latin1')

    x_data = dict_data['data']
    t_data = dict_data['labels']

    N = x_data.shape[0]

    x_data = x_data.reshape(N, 3, 32, 32).transpose(0, 2, 3, 1)
    t_onehot = np.zeros((N, 10))
    t_onehot[np.arange(N), t_data] = 1

    return x_data, t_onehot


def get_data():

    FILES_TRAIN = ['./cifar-10-batches-py/data_batch_{}'.format(k)
                   for k in range(1, 6)]
    FILE_TEST = './cifar-10-batches-py/test_batch'

    x_test, t_test = process_data(FILE_TEST)

    x_list = []
    t_list = []
    for file_name in FILES_TRAIN:
        x_data, t_data = process_data(file_name)
        x_list.append(x_data)
        t_list.append(t_data)

    x_data = np.vstack(x_list)
    t_data = np.vstack(t_list)

    return x_data, x_test, t_data, t_test


if __name__ == '__main__':

    maybe_download_and_extract()
    x_data, x_test, t_data, t_test = get_data()

    layer_list = []

    layer =  {'dim': (32, 32, 3)}
    layer_list.append(layer)

    layer = {'type': 'conv',
             'channels': 64,
             'k_size': (5, 5),
             'strides': (1, 1),
             'padding': 'SAME',
             'activation': 'relu',
             'init_w': 'truncated_normal',
             'stddev_w': 1.0,
             'init_b': 'zeros'}
    layer_list.append(layer)

    layer = {'type': 'maxpool',
             'k_size': (3, 3),
             'strides': (2, 2),
             'padding': 'SAME'}
    layer_list.append(layer)

    layer = {'type': 'LRN',
             'r': 4,
             'k': 1.0,
             'alpha': 0.001 / 9.0,
             'beta': 0.75}
    layer_list.append(layer)

    layer = {'type': 'conv',
             'channels': 64,
             'k_size': (5, 5),
             'strides': (1, 1),
             'padding': 'SAME',
             'activation': 'relu',
             'init_w': 'truncated_normal',
             'stddev_w': 1.0,
             'init_b': 'zeros'}
    layer_list.append(layer)

    layer = {'type': 'LRN',
             'r': 4,
             'k': 1.0,
             'alpha': 0.001 / 9.0,
             'beta': 0.75}
    layer_list.append(layer)

    layer = {'type': 'maxpool',
             'k_size': (3, 3),
             'strides': (2, 2),
             'padding': 'SAME'}
    layer_list.append(layer)

    layer = {'type': 'fc',
             'dim': 384,
             'activation': 'relu',
             'init_w': 'truncated_normal',
             'stddev_w': 1.0,
             'init_b': 'zeros'}
    layer_list.append(layer)

    layer = {'type': 'fc',
             'dim': 192,
             'activation': 'relu',
             'init_w': 'truncated_normal',
             'stddev_w': 1.0,
             'init_b': 'zeros'}
    layer_list.append(layer)

    layer = {'type': 'fc',
             'dim': 10,
             'activation': 'softmax',
             'init_w': 'truncated_normal',
             'stddev_w': 1.0,
             'init_b': 'zeros'}
    layer_list.append(layer)

    nnet = NetConstructor(layer_list)

    TRAIN = True
    if TRAIN:
        nnet.train(x_data, t_data,
                   method=('adam', {'eta': 0.001,
                                    'beta_1': 0.9,
                                    'beta_2': 0.999,
                                    'epsilon': 1e-8}),
                   nb_epochs=350,
                   batch_size=128, print_cost=False)

    pred = nnet.predict(x_test)

    true_classes = np.argmax(t_test, axis=1)
    pred_classes = np.argmax(pred, axis=1)


    nb_good_pred = np.sum(np.equal(true_classes, pred_classes))
    print(nb_good_pred / x_test.shape[0]) # aprox 0.64 %

    # sugerencias:
    # - añadir más capas convolucionales
    # - normalización de los datos o, mejor:
    # - añadir capas de batch normalization (y posiblemente suprimir dropouts)
    # - considerar mejores inicializaciones
    # - considerar activaciones mejores que relu
    # - implementar mejores métodos de parada (early stopping)
    # - considerar otros tamaños de máscaras de convolución
