# Restrict Boltzmann Machine 
import tensorflow as tf 
import tensorflow.compat.v1 as tf1
import math
import timeit
import numpy as np 
import matplotlib.pyplot as plt
from Shared.utils import tile_raster_images


def weight_variable(shape):
    initial = tf1.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape = shape)
    return tf.Variable(initial)

class GRBM(object):
	# Gaussian Restrict Boltzmann Machine
	def __init__(self, inp = None, n_visible = 784, n_hidden = 500, W = None, hbias = None, vbias = None, sigma = 1.0):
		
		self.n_visible = n_visible
		self.n_hidden = n_hidden
		if inp is None:
			inp = tf.placeholder(dtype = tf.float32, shape = [None, self.n_visible])
		self.input = inp

		
		if W is None:
			low = -4.0 * math.sqrt(6. / (n_visible+ n_hidden))
			high = 4.0 * math.sqrt(6. / (n_visible + n_hidden))
			W = tf.Variable(tf.random_uniform([n_visible, self.n_hidden], minval = low, maxval = high, dtype = tf.float32))

		W = weight_variable([n_visible, self.n_hidden])
		
		self.W = W
		if hbias is None:
			hbias = tf.Variable(tf.zeros([n_hidden]), dtype = tf.float32)
		self.hbias = hbias
	
		if vbias is None:
			vbias = tf.Variable(tf.zeros([n_visible]), dtype = tf.float32)
		
		self.vbias = vbias
		#super(GRBM, self).__init__(inp, n_visible, n_hidden, W, hbias, vbias)
		self.sigma = sigma
		self.params = [self.W, self.hbias, self.vbias]


	def propup(self, visible):
		#print(visible.shape)
		#print(self.W)
		# This function propagates the visible units activation upwards to the hidden unit
		return tf.nn.sigmoid(tf.matmul(visible, self.W) / (self.sigma **2) + self.hbias)	

	def propdown(self, hidden):
		# This function propagates the hidden units activaion downwards to the visible unit
		#dist = tf.contrib.distributions.Normal(loc = tf.matmul(hidden, tf.transpose(self.W)) + self.vbias, scale = self.sigma **2)
		pre =  tf.matmul(hidden, tf.transpose(self.W)) + self.vbias
		#return dist.prob(tf.matmul(hidden, tf.transpose(self.W)) + self.vbias)
		return (pre, tf.nn.sigmoid(pre))

	def sample_bernoulli(self, prob):
		return tf.nn.relu(tf.sign(prob - tf1.random_uniform(tf1.shape(prob))))

	def sample_gaussian(self, x, sigma):
		return x + tf1.random_normal(tf.shape(x), mean = 0.0, stddev = sigma, dtype=tf.float32)


	def sample_h_given_v(self, v0_sample):
		# This function infers state of hidden units given visible units
		# get a sample of the hiddens given their activation
		h1_mean = self.propup(v0_sample)
		h1_sample = self.sample_bernoulli(h1_mean)
		return (h1_mean, h1_sample)

	def sample_v_given_h(self, h0_sample):
		# This function infers state of visible units given hidden units
		# get a sample of the hiddens given their activation
		pre, v1_mean = self.propdown(h0_sample)
		v1_sample = self.sample_gaussian(v1_mean, self.sigma)
		v1_sample = pre
		return (v1_mean, v1_sample)

		'''gibbs_vhv which performs a step of Gibbs sampling starting from the visible units. 
		As we shall see, this will be useful for sampling from the RBM.
		gibbs_hvh which performs a step of Gibbs sampling starting from the hidden units. 
		This function will be useful for performing CD and PCD updates.'''

	def gibbs_hvh(self, h_sample):
		# this function implements one step of Gibbs sampling starting from the hidden state
		v1_mean, v1_sample = self.sample_v_given_h(h_sample)
		h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
		return (v1_mean, v1_sample, h1_mean, h1_sample)

	def gibbs_vhv(self, v_sample):
		# this function implements one step of gibbs sampling starting from the visible state
		h1_mean, h1_sample = self.sample_h_given_v(v_sample)
		v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
		return (h1_mean, h1_sample, v1_mean, v1_sample)

	def free_energy(self, v_sample):
		#function to compute the free energy which need for computing the gradient
		wx_b = tf.matmul(v_sample, self.W) / self.sigma**2 + self.hbias
		vbias_term = tf1.reduce_sum(0.5*tf.square(v_sample - self.vbias)/ (self.sigma**2), axis = 1)
		hidden_term = tf1.reduce_sum(tf1.log(1.0 + tf.exp(wx_b)), axis =1 )
		return -hidden_term + vbias_term
	
	#we then add a train_ops method, whose purpose is to generate the sysbolic gradients from CD-k and PCD-k updates
	def get_train_ops(self, lr = 0.01, persistent = None, k = 1):
		''' This function implements one step of CD-k or PCD-k
		:param lr: leaning rate used to train the rbm
		:param persistent: none for Cd, for PCD, shared variable containing old state of
		Gibbs chain. This must be a shared variable of size(batch_size), number of hidden units)
		:param k : number if Gibbs step to do in CD-k, PCD-k

		Return a proxy for the cost and the updates dictionary.
		The dictionary contains the updates rules for the weights and biases
		but also an update of the shared variable used to store the persistent
		chain, if one is used'''

		# compute positive phase
		ph_mean, ph_sample = self.sample_h_given_v(self.input)
		#decide how to initialize persistent chain:
		#for cd, we use the newly generate hidden sample
		# for PCD, we initialize from the old state of the chain

		if persistent is None:
			chain_start = ph_sample
		else:
			chain_start = persistent

		#perform actual negative phase
		#in order to implement CD-k/ PCd-k we need to scan over the function that implement one gibbs step k times

		cond = lambda i, nv_mean, nv_sample, nh_mean, nh_sample: i < k
		body = lambda i, nv_mean, nv_sample, nh_mean, nh_sample: (i+1, ) + self.gibbs_hvh(nh_sample)
		
		i, nv_mean, nv_sample, nh_mean, nh_sample = tf.while_loop(cond, body, loop_vars = [tf.constant(0), tf.zeros(tf.shape(self.input)), tf.zeros(tf.shape(self.input)), tf.zeros(tf.shape(chain_start)), chain_start])
		# determine gradients on RBM parameters

		# note that we only need the sample at the end of the chain
		chain_end = tf.stop_gradient(nv_sample)
		#mean, v_sample = self.sample_v_given_h(ph_sample)
		self.cost = tf.reduce_mean(self.free_energy(self.input)) - tf.reduce_mean(self.free_energy(chain_end))

		#we must not compute the gradient through the gibbs sampling
		# compute the gradients
		gparams = tf.gradients(ys = [self.cost], xs = self.params)
		new_params = []
		for gparam, param in zip(gparams, self.params):
			new_params.append(tf1.assign(param, param - gparam * lr))
		cost = tf.reduce_mean(tf1.reduce_sum(tf1.square(self.input - nv_mean), axis=1))
			#cost = -tf.reduce_mean(tf.reduce_sum(self.input * tf.log(nv_mean) + (1.0 - self.input) * tf.log(1.0 - nv_mean), axis = 1))
		if persistent is not None:
			new_persistent = [tf1.assign(persistent, nh_sample)]
		else:
			new_persistent = []
		print("grbm")
		return cost,new_params + new_persistent
	

	def get_reconstruction_cost(self):
		#compute the cross-entropy of the original inout and the reconstruction
		act_h = self.propup(self.input)
		_, act_v = self.propdown(act_h)
		#print(act_h.shape)
		#print(act_v)
		#cross_entropy = -tf.reduce_mean(tf.reduce_sum(self.input * tf.log(act_v) + (1.0 - self.input) * tf.log(1.0 - act_v), axis = 1))
		cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(self.input - act_v), axis=1))
		print("cost grbm")
		return cross_entropy
	
	def reconstruction(self, inp):
		act_h = self.propup(inp)
		return self.propdown(act_h)


class RBM(object):
	"""A Restricted Boltzmann Machines class"""
	def __init__(self, inp = None, n_visible = 784, n_hidden = 500, W = None, hbias = None, vbias = None):
		self.n_visible = n_visible
		self.n_hidden = n_hidden
		if inp is None:
			inp = tf.placeholder(dtype = tf.float32, shape=[None, self.n_visible])
		self.input = inp
		if W is None:
			low = -4.0 * math.sqrt(6. / (n_visible+ n_hidden))
			high = 4.0 * math.sqrt(6. / (n_visible + n_hidden))
			W = tf.Variable(tf.random_uniform([n_visible,self.n_hidden], minval = low, maxval = high, dtype = tf.float32))
		W = weight_variable([n_visible,self.n_hidden])
		self.W = W
		if hbias is None:
			hbias = tf.Variable(tf.zeros([n_hidden]), dtype = tf.float32)
		self.hbias = hbias
		if vbias is None:
			print(n_visible)
			vbias = tf.Variable(tf.zeros([n_visible]), dtype = tf.float32)
		self.vbias = vbias
		self.params = [self.W, self.hbias, self.vbias]

	def propup(self, visible):
		"""This function propagates the visible units activation upwards to the hidden units"""
		return tf.nn.sigmoid(tf.matmul(visible,self.W) + self.hbias)


	def propdown(self, hidden):
		"""This function propagates the hidden units activation downwards to the visible units"""
		return tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(self.W)) + self.vbias)

	def sample_prob(self, prob):
		'''Do sampling with the given probability'''
		return tf.nn.relu(tf.sign(prob - tf1.random_uniform(tf.shape(prob))))
	
	def sample_h_given_v(self, v0_sample):
		''' This function infers state of hidden units given visible units'''
		# get a sample of the hiddens given their activation
		h1_mean = self.propup(v0_sample)
		h1_sample = self.sample_prob(h1_mean)
		return (h1_mean, h1_sample)

	def sample_v_given_h(self, h0_sample):
		''' This function infers state of visible units given hidden units '''
		# get a sample of the visible given their activation
		v1_mean = self.propdown(h0_sample)
		v1_sample = self.sample_prob(v1_mean)
		return (v1_mean, v1_sample)

		'''gibbs_vhv which performs a step of Gibbs sampling starting from the visible units. 
    	As we shall see, this will be useful for sampling from the RBM.
		gibbs_hvh which performs a step of Gibbs sampling starting from the hidden units. 
		This function will be useful for performing CD and PCD updates.'''

	def gibbs_hvh(self, h_sample):
		'''This function implements one step of Gibbs sampling,
            starting from the hidden state'''
		v1_mean, v1_sample = self.sample_v_given_h(h_sample)
		h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
		return (v1_mean, v1_sample, h1_mean, h1_sample)


	def gibbs_vhv(self, v_sample):
		''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
		h1_mean, h1_sample = self.sample_h_given_v(v_sample)
		v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
		return (h1_mean, h1_sample, v1_mean, v1_sample)


	def free_energy(self, v_sample):
		'''function to compute the free energy which need for
    		computing the gradient of the parameters'''
		wx_b = tf.matmul(v_sample, self.W) + self.hbias
		vbias_term = tf.matmul(v_sample, tf.expand_dims(self.vbias, axis = 1))
		hidden_term = tf1.reduce_sum(tf1.log(1 + tf1.exp(wx_b)), axis = 1)
		return -hidden_term - vbias_term

	# we then add a get_train_ops method, whose purpose is to generate the
	# symbolic gradients forn CD-k and PCD-k updates

	def get_train_ops(self, lr = 0.001, persistent = None, k = 1):
		'''This functions implements one step of CD-k or PCD-k

      :param lr: learning rate used to train the RBM
      :param persistent: None for CD. For PCD, shared variable containing 
      old state of Gibbs chain. This must be a shared variable of size(batch
      size, number of hidden units)
      :param k: number of Gibbs step to do in CD-k/PCD-k

    	Return a proxy for the cost and the updates dictionary. 
      The dictionary contains the update rules for weights and biases
      but also an update of the shared variable used to store the persistent
      chain, if one is used.'''
		
		#compute positive phase

		ph_mean, ph_sample = self.sample_h_given_v(self.input)

		#decide how to initialize persistent chain:
		# for CD, we use the newly generate hidden sample
		#forn PCD, we initialize from the old state of the chain
		if persistent is None:
			chain_start = ph_sample
		else:
			chain_start = persistent
		# perform actual negative phase
		# in order to implement CD-k/ PCD-k we need to scan over the
		# function that implements one gibbs step k times
		#print( tf.shape(chain_start))
		cond = lambda i, nv_mean, nv_sample, nh_mean, nh_sample: i < k
		body = lambda i, nv_mean, nv_sample, nh_mean, nh_sample: (i + 1, ) + self.gibbs_hvh(nh_sample)
		i, nv_mean, nv_sample, nh_mean, nh_sample = tf.while_loop(cond, body, loop_vars=[tf.constant(0), tf.zeros(tf.shape(self.input)), tf.zeros(tf.shape(self.input)), tf.zeros(tf.shape(chain_start)), chain_start])
		# determine gradients on RBM parameters

		# note that we only need the sample at the end of the chain
		chain_end = tf.stop_gradient(nv_sample)

		self.cost = tf.reduce_mean(self.free_energy(self.input)) - tf.reduce_mean(self.free_energy(chain_end))
		# We must not compute the gradient through the gibbs sampling
		#compute the gradients
		gparams = tf.gradients(ys = [self.cost], xs = self.params)
		new_params = []
		for gparam, param in zip(gparams, self.params):
			new_params.append(tf1.assign(param, param - gparam * lr))
		cost = tf1.reduce_mean(tf1.reduce_sum(tf1.square(self.input - nv_mean), axis=1))
		if persistent is not None:
			new_persistent = [tf1.assign(persistent, nh_sample)]
		else:
			new_persistent = []
		print("rbm")
		return cost, new_params + new_persistent
		
	def get_reconstruction_cost(self):
		'''compute the cross-entropy of the original input and the reconstruction'''
		act_h = self.propup(self.input)
		act_v = self.propdown(act_h)
		cross_entropy = -tf1.reduce_mean(tf1.reduce_sum(self.input * tf1.log(act_v) + (1.0 - self.input)* tf1.log(1.0 - act_v), axis = 1))
		print("cost rbm")
		return cross_entropy
		"""
	def get_reconstruction_cost(self):
		'''Compute the cross-entropy of the original input and the reconstruction'''
		activation_h = self.propup(self.input)
		activation_v = self.propdown(activation_h)
		# Do this to not get Nan
		activation_v_clip = tf.clip_by_value(activation_v, clip_value_min=1e-30, clip_value_max=1.0)
		reduce_activation_v_clip = tf.clip_by_value(1.0 - activation_v, clip_value_min=1e-30, clip_value_max=1.0)
		cross_entropy = -tf.reduce_mean(tf.reduce_sum(self.input*(tf.log(activation_v_clip)) + 
                                    (1.0 - self.input)*(tf.log(reduce_activation_v_clip)), axis=1))
		return cross_entropy   
		"""


	def reconstruction(self, v):
		h = self.propup(v)
		return self.propdown(h)
