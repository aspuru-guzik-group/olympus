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

import numpy as np 
import multiprocessing
from multiprocessing import Manager, Process

from utilities import Logger

#========================================================================

class SampleSelector(Logger):

	def __init__(self, config):
		self.config = config
		Logger.__init__(self, 'SampleSelector', verbosity = self.config.get('verbosity'))
		self.num_cpus = multiprocessing.cpu_count()

	def compute_exp_objs(self, proposals, bayesian_network, batch_index, return_index, result_dict = None):

		samples  = proposals[batch_index]
		exp_objs = np.empty(len(samples))

		for sample_index, sample in enumerate(samples):
			num, inv_den = bayesian_network.kernel_contribution(sample)
			kernel_contrib = (num + self.sampling_param_values[batch_index]) * inv_den
			exp_objs[sample_index] = np.exp( - kernel_contrib)

		if result_dict.__class__.__name__ == 'DictProxy':
			result_dict[return_index] = exp_objs
		else:
			return exp_objs


	
	def select(self, num_samples, proposals, bayesian_network, sampling_param_values, obs_params):

		num_obs = len(obs_params)	
		feature_ranges = self.config.feature_ranges
		char_dists     = feature_ranges / float(num_obs)**0.5
		self.sampling_param_values = sampling_param_values		
	
		# compute acq func values
		if self.config.get('parallel'):
			result_dict = Manager().dict()
			
			# get the number of splits
			num_splits = self.num_cpus // len(sampling_param_values) + 1
			split_size = proposals.shape[1] // num_splits

			processes = []
			for batch_index in range(len(sampling_param_values)):
				for split_index in range(num_splits):
					
					split_start = split_size * split_index
					split_end   = split_size * (split_index + 1)
					return_index = num_splits * batch_index + split_index
					process = Process(target = self.compute_exp_objs, args = (proposals[:, split_start : split_end], bayesian_network, batch_index, return_index, result_dict))
					processes.append(process)
					process.start()
				for process_index, process in enumerate(processes):
					process.join()

		else:
			num_splits  = 1
			result_dict = {}
			for batch_index in range(len(sampling_param_values)):
				return_index = batch_index
				result_dict[return_index] = self.compute_exp_objs(proposals, bayesian_network, batch_index, return_index)

		# collect results
		exp_objs = []
		for batch_index in range(len(sampling_param_values)):
			batch_exp_objs = []
			for split_index in range(num_splits):
				return_index = num_splits * batch_index + split_index

				batch_exp_objs.append(result_dict[return_index])
			exp_objs.append(np.concatenate(batch_exp_objs))
		exp_objs = np.array(exp_objs)

		# compute prior recommendation punishments
		for batch_index in range(len(sampling_param_values)):
			batch_proposals = proposals[batch_index, : exp_objs.shape[1]]

			# compute distance to each obs_param
			distances     = [ np.sum((obs_params - batch_proposal)**2, axis = 1) for batch_proposal in batch_proposals]
			distances     = np.array(distances)			
			min_distances = np.amin(distances, axis = 1)
			ident_indices = np.where(min_distances < 1e-8)[0]
		
			exp_objs[batch_index, ident_indices] = 0.


		# collect samples
		samples = []
		for sample_index in range(num_samples):
			new_samples = []

			for batch_index in range(len(sampling_param_values)):
				batch_proposals = proposals[batch_index]

				# compute diversity punishments
				div_crits = np.ones(exp_objs.shape[1])

				for proposal_index, proposal in enumerate(batch_proposals[:exp_objs.shape[1]]):

					obs_min_distance = np.amin([np.abs(proposal - x) for x in obs_params], axis = 0)
					if len(new_samples) > 0:
						min_distance = np.amin([np.abs(proposal - x) for x in new_samples], axis = 0)
						min_distance = np.minimum(min_distance, obs_min_distance)
					else:
						min_distance = obs_min_distance

					div_crits[proposal_index] = np.minimum(1., np.mean( np.exp( 2. * (min_distance - char_dists) / feature_ranges) ))					


				# reweight rewards
				reweighted_rewards   = exp_objs[batch_index] * div_crits
				largest_reward_index = np.argmax( reweighted_rewards )
				
				new_sample = batch_proposals[largest_reward_index]
				new_samples.append(new_sample)

				# update reward of selected sample
				exp_objs[batch_index, largest_reward_index] = 0.

			samples.append(new_samples)
		samples = np.concatenate(samples)
		return samples






