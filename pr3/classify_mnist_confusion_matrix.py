from __future__ import division, print_function

import time

import cPickle
import gzip
import sys
import numpy as np
import mlp
reload(mlp)
from mlp import MLP
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix


# Found at: http://scikit-learn.org/stable/auto_examples/model_selection/...
# plot_confusion_matrix.html

"""
We have decided to plot a confusion matrix and try to see where we get the
mispredictions. The documentation regarding this function can be found in the
link before
"""


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

x_data = train_set[0]
x_data = x_data
mean_image = np.mean(x_data, axis=0)
x_data -= mean_image
x_data = x_data / 255

t_data = train_set[1]
nb_data = t_data.shape[0]
one_hot_tdata = np.zeros((nb_data, 10))
one_hot_tdata[np.arange(nb_data), t_data] = 1


K_list = [784, 100, 50, 10]
activation_functions = [MLP.relu] * 2 + [MLP.sigmoid]

diff_activation_functions = [MLP.drelu] * 2


mlp = MLP(K_list,
          activation_functions,
          diff_activation_functions,
          init_seed=5)

if 1:
    x_test, t_test = test_set
    nb_epochs = 250

    for epoch in range(nb_epochs):
        initialize_weights = (epoch == 0)
        now = time.time()
        mlp.train(x_data, one_hot_tdata,
                  epochs=1,
                  batch_size=60,
                  initialize_weights=initialize_weights,
                  eta=0.01,
                  beta=0,
                  method='adam',
                  print_cost=True)
        time_passed = time.time() - now
        print(time_passed)

        mlp.get_activations_and_units((x_test - mean_image) / 255)
        nb_correct = np.sum(np.equal(t_test, np.argmax(mlp.y, axis=1)))
        sys.stdout.write('aciertos en los datos de test = %d, epoch= %d\n'
                         % (nb_correct, epoch))
        sys.stdout.write('fallos en los datos de test = %d, epoch= %d\n'
                         % (10000 - nb_correct, epoch))
        sys.stdout.write('porcentaje de aciertos en los datos de test = %d\n'
                         % ((nb_correct / 10000) * 100))
        sys.stdout.flush()

    np.save('mnist_weights_v2', mlp.weights_list, allow_pickle=True)
    np.save('mnist_biases_v2', mlp.biases_list, allow_pickle=True)

    print(nb_correct)
    
if 0:
    np.load('./mnist_weights_v2.npy')
    np.load('./mnist_biases_v2.npy')

    x_test, t_test = test_set
    #mean_image = np.mean(x_test, axis=0)
    mlp.get_activations_and_units((x_test - mean_image) / 255)
    nb_correct = np.sum(np.equal(t_test, np.argmax(mlp.y, axis=1)))
    print(nb_correct)



class_names = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

# Compute confusion matrix
cnf_matrix = confusion_matrix(t_test, np.argmax(mlp.y, axis=1))


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')


plt.show()
