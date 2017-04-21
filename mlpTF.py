
from __future__ import print_function

import tensorflow as tf
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

#  Fully connected layer 
def fc_layer(z, n_neurons, activ_function, reg_function):
    n_inputs = int(z.get_shape()[1])
    stddev = 1 / np.sqrt(n_inputs)
    w_shape = (n_inputs,n_neurons)
    w_init = tf.truncated_normal(w_shape, stddev=stddev)
    w = tf.Variable(w_init, name="weights")
    
    b_init = tf.zeros([n_neurons])  
    b = tf.Variable(b_init, name='biases')
    
    a = tf.matmul(z, w) + b
    
    reg_loss = 0
    if reg_function is not None:
        reg_loss = reg_function(w)
        
    return activ_function(a), reg_loss


#  Reset del grafo
tf.reset_default_graph()

# Lectura del dataset
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# Parameters
learning_rate = 0.001
training_epochs = 300
batch_size = 100
display_step = 1



# Network Parameters

hidden_list = [256, 256, n_classes]
activ_functions = [tf.nn.relu]*(len(hidden_list)-1) + [tf.identity]
regularizers = [None]*len(hidden_list)
#regularizers = [tf.nn.l2_loss]*len(hidden_list)
dropouts = [0.5,0.5] + [None]
beta = 0



# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

reg_cost = 0
reg_loss = 0

nb_hidden = len(hidden_list)
z = x
for layer in range(nb_hidden):
    z, reg_loss = fc_layer(z,hidden_list[layer],activ_functions[layer],regularizers[layer])
    reg_cost += reg_loss
    if dropouts[layer] is not None:
        z = tf.nn.dropout(z, dropouts[layer])


out = tf.nn.softmax_cross_entropy_with_logits(logits=z,labels=y)
cost = tf.reduce_mean(out + beta*reg_cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(z, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

