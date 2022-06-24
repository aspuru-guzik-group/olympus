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
  
__author__ = 'Florian Hase'

#=========================================================================

import numpy as np 
import multiprocessing
from multiprocessing import Process, Manager

from Acquisition   import ParameterOptimizer
from RandomSampler import RandomSampler
from utilities     import Logger

#=========================================================================

class Acquisition(Logger):

	def __init__(self, config):
	
		self.config = config
		Logger.__init__(self, 'Acquisition', self.config.get('verbosity'))
		self.random_sampler   = RandomSampler(self.config.general, self.config.parameters)
		self.total_num_vars   = len(self.config.feature_names)
		self.local_optimizers = None		
		self.num_cpus         = multiprocessing.cpu_count()


	def _propose_randomly(self, best_params, num_samples, dominant_samples = None, num_obs = 1.0):
		# get uniform samples
		if dominant_samples is None:
			uniform_samples = self.random_sampler.draw(num = self.total_num_vars * num_samples)
			perturb_samples = self.random_sampler.perturb(pos=best_params, num = self.total_num_vars * num_samples, scale=0.3/num_obs)
			samples         = np.concatenate([uniform_samples, perturb_samples])
		else:
			dominant_features = self.config.feature_process_constrained
			for batch_sample in dominant_samples:
				uniform_samples = self.random_sampler.draw(num = self.total_num_vars * num_samples // len(dominant_samples))
				perturb_samples = self.random_sampler.perturb(pos=best_params, num = self.total_num_vars * num_samples, scale=0.3/num_obs)
				samples         = np.concatenate([uniform_samples, perturb_samples])
			samples[:, dominant_features] = batch_sample[dominant_features]
		return samples


	def _proposal_optimization_thread(self, proposals, bayesian_network, batch_index, return_index, return_dict = None, dominant_samples = None):
		self.log('starting process for %d' % batch_index, 'INFO')

		def kernel(x):
			num, inv_den = bayesian_network.kernel_contribution(x)
			return (num + self.sampling_param_values[batch_index]) * inv_den 
	
		if dominant_samples is not None:
			ignore = self.config.feature_process_constrained
		else:
			ignore = np.array([False for _ in range(len(self.config.feature_process_constrained))])

		local_optimizer = self.local_optimizers[batch_index]
		local_optimizer.set_func(kernel, ignores = ignore)	

		optimized = []
		for sample_index, sample in enumerate(proposals):
			opt = local_optimizer.optimize(kernel, sample, max_iter = 100)
			optimized.append(opt)
		optimized = np.array(optimized)

		if return_dict.__class__.__name__ == 'DictProxy':
			return_dict[return_index] = optimized
		else:
			return optimized	


	def _optimize_proposals(self, random_proposals, bayesian_network, dominant_samples = None):

		if self.config.get('parallel'):
			result_dict = Manager().dict()

			# get the number of splits
			num_splits = self.num_cpus // len(self.sampling_param_values) + 1
			split_size = len(random_proposals) // num_splits #+ 1

			processes   = []
			for batch_index in range(len(self.sampling_param_values)):
				for split_index in range(num_splits):
					split_start  = split_size * split_index
					split_end    = split_size * (split_index + 1)
					return_index = num_splits * batch_index + split_index
					process = Process(target = self._proposal_optimization_thread, args = (random_proposals[split_start : split_end], bayesian_network, batch_index, return_index, result_dict, dominant_samples))
					processes.append(process)
					process.start()

			for process_index, process in enumerate(processes):
				process.join()	

		else:
			num_splits  = 1
			result_dict = {}
			for batch_index in range(len(self.sampling_param_values)):
				return_index = batch_index
				result_dict[batch_index] = self._proposal_optimization_thread(random_proposals, bayesian_network, batch_index, return_index, dominant_samples = dominant_samples)

		# collect optimized samples
		samples = []
		for batch_index in range(len(self.sampling_param_values)):
			batch_samples = []
			for split_index in range(num_splits):
				return_index = num_splits * batch_index + split_index
				batch_samples.append(result_dict[return_index])
			samples.append(np.concatenate(batch_samples))
		samples = np.array(samples)
		return np.array(samples)


	def propose(self, best_params, bayesian_network, sampling_param_values, num_obs,
				num_samples = 50, 
				parallel = 'True',
				dominant_samples  = None,
				dominant_strategy = None,
			):

		self.local_optimizers = [ParameterOptimizer(self.config) for _ in range(len(sampling_param_values))]
		assert len(self.local_optimizers) == len(sampling_param_values)
		self.sampling_param_values = sampling_param_values

		random_proposals = self._propose_randomly(
				best_params, num_samples, dominant_samples = dominant_samples, num_obs = num_obs
			)

		import time
		start = time.time()
		optimized_proposals = self._optimize_proposals(
				random_proposals, bayesian_network, dominant_samples = dominant_samples,
			)
		end   = time.time()
		print('[TIME]:  ', end - start, '  (optimizing proposals)')

		extended_proposals = np.array([random_proposals for _ in range(len(sampling_param_values))])
		combined_proposals = np.concatenate((extended_proposals, optimized_proposals), axis = 1)

		return combined_proposals

