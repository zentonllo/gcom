
# coding: utf-8

from __future__ import division, print_function
from mlp import MLP
import numpy as np
import gzip
import cPickle
# import time

def digits_to_vec(t):
    size = t.size
    t_one_hot = np.zeros([size, 10])
    t_one_hot[np.arange(size), t] = 1
    return t_one_hot



my_file = gzip.open('mnist.pkl.gz')
result = cPickle.load(my_file)
my_file.close()

train_data, test_data, validation_data = result
train_images, train_labels = train_data

D = train_images.shape[1]
K = 10

K_list = [D, 30, 20, K]
activation_functions = [MLP.sigmoid, MLP.sigmoid, MLP.identity]
diff_activation_functions  = [MLP.dsigmoid,MLP.dsigmoid,MLP.didentity]
x_data = np.asarray(train_images) 
t_data = digits_to_vec(train_labels)


mlp = MLP(K_list,activation_functions, diff_activation_functions)

mlp.train(x_data, t_data,
          epochs=1000, batch_size=10,
          epsilon=0.1,
          beta=0.001,
          print_cost=False)

mlp.get_activations_and_units(x_data)
res = np.argmax(mlp.softmax(mlp.y), axis = 1)


num_wrong = np.sum(np.abs(digits_to_vec(res) - t_data)) / 2
N = train_images.shape[0]
                  
print('Numeros errores finales: {}'.format(num_wrong))
print('Porcentaje de errores: {}'.format((num_wrong/N)*100))