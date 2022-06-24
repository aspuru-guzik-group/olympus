#!/usr/bin/env python

'''
Licensed to the Apache Software Foundation (ASF) under one or more 
contributor license agreements. See the NOTICE file distributed with this 
work for additional information regarding copyright ownership. The ASF 
licenses this file to you under the Apache License, Version 2.0 (the 
"License"); you may not use this file except in compliance with the 
License. You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT 
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the 
License for the specific language governing permissions and limitations 
under the License.

The code in this file was developed at Harvard University (2018) and 
modified at ChemOS Inc. (2019) as stated in the NOTICE file.
'''

__author__  = 'Florian Hase'

#=========================================================================

import warnings
warnings.filterwarnings('ignore')

import pickle
import numpy as np 
import tensorflow as tf
import edward     as ed
from multiprocessing import Process, Manager

import os, sys
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from numpy_graph import NumpyGraph

#=========================================================================

class EdwardNetwork(object):

	def __init__(self, config, model_details):
		self.config        = config
		self.numpy_graph   = NumpyGraph(self.config, model_details)

		self.model_details = model_details
		for key, value in self.model_details.items():
			setattr(self, '_%s' % str(key), value)

		self.feature_size    = len(self.config.kernel_names)
		self.bnn_output_size = len(self.config.kernel_names)
		self.target_size     = len(self.config.kernel_names)

		self.graph = tf.Graph()
		with self.graph.as_default():
			self.sess = tf.InteractiveSession()


	def declare_training_data(self, obs_params, obs_objs):

		assert len(obs_params) == len(obs_objs)
		# categorical features are encoded as single numbers, bnn output is one-hot encoded
		self.num_obs = len(obs_params)
		self.obs_objs = obs_objs

		
		# initialize training features and targets
		self.features = np.zeros((self.num_obs, self.feature_size))
		self.targets  = np.zeros((self.num_obs, self.target_size)) 

		# construct training features
		feature_begin = 0
		feature_sizes = self.config.feature_sizes
		for feature_index, feature_type in enumerate(self.config.feature_types):
			feature_size = feature_sizes[feature_index]
			self.features[:, feature_begin] = obs_params[:, feature_index]		
			feature_begin += feature_size
		self.targets = self.features.copy()

		# rescale features
		lower_rescalings = np.empty(self.feature_size)
		upper_rescalings = np.empty(self.feature_size)
		kernel_uppers, kernel_lowers = self.config.kernel_uppers, self.config.kernel_lowers
		for kernel_index, kernel_name in enumerate(self.config.kernel_names):
			low = kernel_lowers[kernel_index]
			up  = kernel_uppers[kernel_index]
			lower_rescalings[kernel_index] = low #- 0.1 * (up - low)
			upper_rescalings[kernel_index] = up  #+ 0.1 * (up - low)

		self.lower_rescalings = lower_rescalings
		self.upper_rescalings = upper_rescalings

		self.rescaled_features = (self.features - self.lower_rescalings) / (self.upper_rescalings - self.lower_rescalings)
		self.rescaled_targets  = (self.targets  - self.lower_rescalings) / (self.upper_rescalings - self.lower_rescalings)

		self.numpy_graph.declare_training_data(self.rescaled_features)


	def construct_model(self):

		with self.graph.as_default():

			self.sess.close()
			self.sess = tf.InteractiveSession()
			self.sess.as_default()

			self.x = tf.convert_to_tensor(self.rescaled_features,  dtype = tf.float32)
			self.y = tf.convert_to_tensor(self.targets,   dtype = tf.float32)

			# construct preciness
			self.tau_rescaling = np.zeros((self.num_obs, self.bnn_output_size))
			kernel_ranges      = self.config.kernel_ranges
			for obs_index in range(self.num_obs):
				self.tau_rescaling[obs_index] += kernel_ranges
			self.tau_rescaling = self.tau_rescaling**2
	
			# construct weight and bias shapes
			activations = [tf.nn.tanh]
			weight_shapes, bias_shapes = [[self.feature_size, self._hidden_shape]], [[self._hidden_shape]]
			for _ in range(1, self._num_layers - 1):
				activations.append(tf.nn.tanh)
				weight_shapes.append([self._hidden_shape, self._hidden_shape])
				bias_shapes.append([self._hidden_shape])
			activations.append(lambda x: x)
			weight_shapes.append([self._hidden_shape, self.bnn_output_size])
			bias_shapes.append([self.bnn_output_size])

			# construct prior
			self.prior_layer_outputs = [self.x]
			self.priors = {}
			for layer_index in range(self._num_layers):
				weight_shape, bias_shape = weight_shapes[layer_index], bias_shapes[layer_index]
				activation = activations[layer_index]
	
				weight = ed.models.Normal(loc = tf.zeros(weight_shape) + self._weight_loc, scale = tf.zeros(weight_shape) + self._weight_scale)
				bias   = ed.models.Normal(loc = tf.zeros(bias_shape)   + self._bias_loc,   scale = tf.zeros(bias_shape)   + self._bias_scale)
				self.priors['weight_%d' % layer_index] = weight
				self.priors['bias_%d'   % layer_index] = bias
	
				prior_layer_output = activation(tf.matmul(self.prior_layer_outputs[-1], weight) + bias)
				self.prior_layer_outputs.append(prior_layer_output)
					
			self.prior_bnn_output = self.prior_layer_outputs[-1]
			self.prior_tau_normed = ed.models.Gamma(self.num_obs**2 + tf.zeros((self.num_obs, self.bnn_output_size)), tf.ones((self.num_obs, self.bnn_output_size)))
			self.prior_tau        = self.prior_tau_normed / self.tau_rescaling
			self.prior_scale	  = ed.models.Deterministic(1. / tf.sqrt(self.prior_tau))
	
			# construct posterior
			self.post_layer_outputs = [self.x]
			self.posteriors = {}
			for layer_index in range(self._num_layers):
				weight_shape, bias_shape = weight_shapes[layer_index], bias_shapes[layer_index]
				activation = activations[layer_index]
	
				weight = ed.models.Normal(loc = tf.Variable(tf.random_normal(weight_shape)), scale = tf.nn.softplus(tf.Variable(tf.zeros(weight_shape))))
				bias   = ed.models.Normal(loc = tf.Variable(tf.random_normal(bias_shape)),   scale = tf.nn.softplus(tf.Variable(tf.zeros(bias_shape))))

				self.posteriors['weight_%d' % layer_index] = weight
				self.posteriors['bias_%d'   % layer_index] = bias
	
				post_layer_output = activation(tf.matmul(self.post_layer_outputs[-1], weight) + bias)
				self.post_layer_outputs.append(post_layer_output)
	
			self.post_bnn_output = self.post_layer_outputs[-1]				
			self.post_tau_normed = ed.models.Gamma(self.num_obs**2 + tf.Variable(tf.zeros((self.num_obs, self.bnn_output_size))), tf.nn.softplus(tf.Variable(tf.ones((self.num_obs, self.bnn_output_size)))))
			self.post_tau        = self.post_tau_normed / self.tau_rescaling
			self.post_sqrt_tau   = tf.sqrt(self.post_tau)
			self.post_scale	     = ed.models.Deterministic(1. / self.post_sqrt_tau)

			# map bnn output to prediction
			post_kernels = {}
			targets_dict = {}
	
			target_element_index = 0
			kernel_element_index = 0
	
			while kernel_element_index < len(self.config.kernel_names):
						
				kernel_type = self.config.kernel_types[kernel_element_index]
				kernel_size = self.config.kernel_sizes[kernel_element_index]
	
				feature_begin, feature_end = target_element_index, target_element_index + 1 
				kernel_begin, kernel_end   = kernel_element_index, kernel_element_index + kernel_size
	
				prior_relevant = self.prior_bnn_output[:, kernel_begin : kernel_end]
				post_relevant  = self.post_bnn_output[:,  kernel_begin : kernel_end]
							
				target  = self.y[:, kernel_begin : kernel_end]
				lowers, uppers = self.config.kernel_lowers[kernel_begin : kernel_end], self.config.kernel_uppers[kernel_begin : kernel_end]
	
				prior_support = (uppers - lowers) * (1.2 * tf.nn.sigmoid(prior_relevant) - 0.1) + lowers
				post_support  = (uppers - lowers) * (1.2 * tf.nn.sigmoid(post_relevant) - 0.1)  + lowers
	
				prior_predict = ed.models.Normal(prior_support, self.prior_scale[:, kernel_begin : kernel_end])				
				post_predict  = ed.models.Normal(post_support,  self.post_scale[:,  kernel_begin : kernel_end])
	
				targets_dict[prior_predict] = target
				post_kernels['param_%d' % target_element_index] = {
					'loc':       ed.models.Deterministic(post_support),
					'sqrt_prec': ed.models.Deterministic(self.post_sqrt_tau[:, kernel_begin : kernel_end]),
					'scale':     ed.models.Deterministic(self.post_scale[:, kernel_begin : kernel_end])}

				target_element_index += 1
				kernel_element_index += kernel_size

			self.post_kernels = post_kernels
			self.targets_dict = targets_dict
				

			# inference
			inference_dict = {}
			for key, prior_var in self.priors.items():
				inference_dict[prior_var] = self.posteriors[key]

			self.inference = ed.KLqp(inference_dict, data = self.targets_dict)
			self.optimizer = tf.train.AdamOptimizer(self._learning_rate)
			self.inference.initialize(optimizer = self.optimizer, n_iter = self._num_epochs)

			tf.global_variables_initializer().run()



	def sample_thread(self, kernel_value, kernel_name, return_dict):

		with self.graph.as_default():
			print('starting sampling', kernel_name)
			samples = [kernel_value.sample().eval() for _ in range(self._num_draws)]
			return_dict[kernel_name] = np.array(samples)
			print('completed sampling')



	def sample(self, num_epochs = None, num_draws = None):
		if num_epochs is None: num_epochs = self._num_epochs
		if num_draws  is None: num_draws  = self._num_draws
		
		with self.graph.as_default():

			# run inference
			import time 
			start = time.time()
			for _ in range(num_epochs):
				info_dict = self.inference.update()
				self.inference.print_progress(info_dict)
			self.inference.finalize()
			end = time.time()
			print('[TIME]:  ', end - start, '  (inference)')

			# sample posterior
			self.trace = {}
			posterior_samples = {}
			for key, kernel_parent in self.posteriors.items():
				parent_samples = kernel_parent.sample(self._num_draws).eval()
				posterior_samples[key] = parent_samples
	

			import time 
			start        = time.time()	
			post_kernels = self.numpy_graph.compute_kernels(posterior_samples)
			end          = time.time()
			print('[TIME]:  ', end - start, '  (sampling from posterior)')


			for key in post_kernels.keys():
				self.trace[key] = {}
				kernel_dict = post_kernels[key]
				for kernel_name, kernel_values in kernel_dict.items():
					self.trace[key][kernel_name] = kernel_values


	def get_kernels(self):		
		trace_kernels = {'locs': [], 'sqrt_precs': []}				
		for param_index in range(len(self.config.param_names)):
			post_kernel = self.trace['param_%d' % param_index]

			trace_kernels['locs'].append(post_kernel['loc'].astype(np.float64))
			trace_kernels['sqrt_precs'].append(post_kernel['sqrt_prec'].astype(np.float64))

		for key, kernel in trace_kernels.items():
			trace_kernels[key] = np.concatenate(kernel, axis = 2)

		return trace_kernels, self.obs_objs


#=========================================================================

if __name__ == '__main__':

	import sys
	home_path    = sys.argv[1]
	sim_file     = sys.argv[2]
	results_file = sys.argv[3]

	sys.path.append(home_path)	

	with open(sim_file, 'rb') as content:
		sim_data = pickle.load(content)

	edward_network = EdwardNetwork(sim_data['config'], sim_data['model_details'])
	edward_network.declare_training_data(sim_data['obs_params'], sim_data['obs_objs'])
	edward_network.construct_model()
	edward_network.sample()

	trace_kernels, obs_objs = edward_network.get_kernels()
	results = {'trace_kernels': trace_kernels, 'obs_objs': obs_objs}
	with open(results_file, 'wb') as content:
		pickle.dump(results, content)


















