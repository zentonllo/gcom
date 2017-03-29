# %%
    # training method for the neuron
    def train(self, x_data, t_data,
              epochs, batch_size,
              initialize_weights=False,
              epsilon=0.01,
              beta=0,
	      gamma=0.9,
              print_cost=False,
	      method='SGD'):

"""
+ añadimos parametro method
- falta añadir los parámetros concretos para cada método (momentum(gamma))

"""






	def SGD():
		self.get_gradients(x_data[indexes], t_data[indexes], beta)
		self.weights_list = [(self.weights_list[k] -
	                             epsilon*self.grad_w_list[k])
	                             for k in range(self.nb_layers)]
		self.biases_list = [(self.biases_list[k] -
                            	    epsilon*self.grad_b_list[k])
                            	    for k in range(self.nb_layers)]
	
% añadir gamma a los atributos
 

dictionary = {'SGD' : {'name' : SGD, 'params' : [,]},
	      'momentum' : {'name': momentum, 'params' : [,]}}


"""
INICIALIZACIÓN DE PARÁMETROS PARA MOMENTUM, HAY QUE HACERLO FUERA DE TRAIN
	for k in range(self.nb_layers):
			dictionary[method]['params'][k] = (np.zeros(self.K_list[layer], 
								   self.K_list[layer + 1]),
							   np.zeros(self.K_list[layer + 1]))
"""

	def momentum(v):
		self.get_gradients(x_data[indexes], t_data[indexes], beta)

		for k in range(self.nb_layers):
			v_w, v_b = dictionary[method]['params'][k]
			v_w = gamma * v_w + epsilon*self.grad_w_list[k]
			v_b = gamma * v_b + epsilon*self.grad_b_list[k]
			dictionary[method]['params'][k] = v_w, v_b

			self.weights_list[k] = (self.weights_list[k] -
	                            	        epsilon*self.grad_w_list[k])
			self.biases_list[k] = (self.biases_list[k] -
                            	               epsilon*self.grad_b_list[k])

		
		
		

        if initialize_weights:
            self.init_weights()

        nb_data = x_data.shape[0]
        index_list = np.arange(nb_data)
        nb_batches = int(nb_data / batch_size)

        for _ in range(epochs):
            np.random.shuffle(index_list)
		for batch in range(nb_batches):
                indexes = index_list[batch*batch_size:(batch+1)*batch_size]
            	

	dictionary[method]['name']()
"""
		if (method=='SGD')
			SGD()
	    	else if (method=='momentum')
			momentum(gamma)
"""



                

            if print_cost:
                x_batch = x_data
                t_batch = t_data
                self.get_activations_and_units(x_batch)
                if self.activation_functions[-1] == MLP.sigmoid:
                    sys.stdout.write('cost = %f\r' %
                                     MLP.binary_cross_entropy(self.y, t_batch))
                    sys.stdout.flush()
                elif self.activation_functions[-1] == MLP.softmax:
                    sys.stdout.write('cost = %f\r' %
                                     MLP.softmax_cross_entropy(
                                         self.y, t_batch))
                    sys.stdout.flush()
                else:
                    sys.stdout.write('cost = %f\r' %
                                     MLP.cost_L2(self.y, t_batch))
                    sys.stdout.flush()
