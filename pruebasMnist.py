
# coding: utf-8

# In[59]:

get_ipython().magic(u'matplotlib tk')

from __future__ import division, print_function
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


# In[46]:

import gzip
import cPickle


# In[47]:

my_file = gzip.open('mnist.pkl.gz')


# In[48]:

result = cPickle.load(my_file)


# In[49]:

type(result)


# In[50]:

len(result)


# In[51]:

train_data, test_data, validation_data = result


# In[52]:

type(train_data)


# In[53]:

train_images, train_labels = train_data


# In[ ]:




# In[54]:

train_images.shape


# In[ ]:




# In[22]:




# In[42]:

img = train_images[0].reshape(28,28)


# In[ ]:




# In[25]:




# In[56]:

plt.imshow(img, cmap="gray")


# In[44]:




# In[57]:

np.sqrt(train_images.shape[1])


# In[58]:

train_labels


# In[66]:

def digits_to_vec(t):
    size = t.size
    t_one_hot = np.zeros([size, 10])
    t_one_hot[np.arange(size), t] = 1
    return t_one_hot


# In[67]:

digits_to_vec(np.array([2,3,0]))


# In[70]:

from mlp import MLP
import time


# In[77]:

K_list = [train_images.shape[1], 10, 10, 10]
activation_functions = [MLP.sigmoid, MLP.sigmoid, MLP.softmax]
diff_activation_functions  = [MLP.dsigmoid,MLP.dsigmoid,MLP.didentity]
x_data = np.asarray(train_images) 
t_data = digits_to_vec(train_labels)


mlp = MLP(K_list,
          activation_functions, diff_activation_functions)



# In[74]:

time_end - time_begin


# In[78]:

mlp.train(x_data, t_data,
          epochs=10, batch_size=10,
          epsilon=0.1,
          beta=0.001,
          print_cost=True)

mlp.softmax_cross_entropy(mlp.y, t_data)


# In[79]:

mlp.softmax_cross_entropy(mlp.y, t_data)


# In[ ]:



