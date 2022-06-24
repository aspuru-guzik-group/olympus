#!/usr/bin/env 

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

import numpy as np 

#=========================================================================

def sigmoid(x):
	return 1. / (1. + np.exp( - x))

#=========================================================================

class NumpyGraph(object):

	def __init__(self, config, model_details):
		self.config = config

		self.model_details = model_details
		for key, value in self.model_details.items():
			setattr(self, '_%s' % str(key), value)

		self.feature_size    = len(self.config.kernel_names)
		self.bnn_output_size = len(self.config.kernel_names)
		self.target_size     = len(self.config.kernel_names)

	
	def declare_training_data(self, features):
		
		self.num_obs  = len(features)
		self.features = features


	def compute_kernels(self, posteriors):
	
		tau_rescaling = np.zeros((self.num_obs, self.bnn_output_size))
		kernel_ranges = self.config.kernel_ranges
		for obs_index in range(self.num_obs):
			tau_rescaling[obs_index] += kernel_ranges
		tau_rescaling = tau_rescaling**2
	
		# sample from BNN
		activations = [np.tanh, np.tanh, lambda x: x]
#		post_layer_outputs = [self.features]
		post_layer_outputs = [np.array([self.features for _ in range(self._num_draws)])]
		for layer_index in range(self._num_layers):

			weight     = posteriors['weight_%d' % layer_index]
			bias       = posteriors['bias_%d'   % layer_index]
			activation = activations[layer_index]

			outputs = []
			for sample_index in range(len(weight)):
				single_weight = weight[sample_index]
				single_bias   = bias[sample_index]

#				output = activation( np.matmul( post_layer_outputs[-1], single_weight) + single_bias)
				output = activation( np.matmul( post_layer_outputs[-1][sample_index], single_weight) + single_bias)
				outputs.append(output)


			post_layer_output = np.array(outputs)
			post_layer_outputs.append(post_layer_output)

		post_bnn_output = post_layer_outputs[-1]

		# note: np.random.gamma is parametrized with k and theta, while ed.models.Gamma is parametrized with alpha and beta
		post_tau_normed = np.random.gamma( self.num_obs**2 + np.zeros(post_bnn_output.shape), np.ones(post_bnn_output.shape))
		post_tau        = post_tau_normed / tau_rescaling
		post_sqrt_tau   = np.sqrt(post_tau)
		post_scale      = 1. / post_sqrt_tau

		# map BNN output to predictions
		post_kernels = {}
		
		target_element_index = 0
		kernel_element_index = 0

		while kernel_element_index < len(self.config.kernel_names):
		
			kernel_type = self.config.kernel_types[kernel_element_index]
			kernel_size = self.config.kernel_sizes[kernel_element_index]

			feature_begin, feature_end = target_element_index, target_element_index + 1
			kernel_begin,  kernel_end  = kernel_element_index, kernel_element_index + kernel_size

			post_relevant = post_bnn_output[:, :, kernel_begin : kernel_end]

			lowers       = self.config.kernel_lowers[kernel_begin : kernel_end]
			uppers       = self.config.kernel_uppers[kernel_begin : kernel_end]
			post_support = (uppers - lowers) * (1.2 * sigmoid(post_relevant) - 0.1) + lowers
				
			post_kernels['param_%d' % target_element_index] = {'loc':       post_support,
															   'sqrt_prec': post_sqrt_tau[:, :, kernel_begin : kernel_end],
															   'scale':     post_scale[:, :, kernel_begin : kernel_end]}

			target_element_index += 1
			kernel_element_index += kernel_size

		return post_kernels



