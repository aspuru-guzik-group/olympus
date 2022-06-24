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

#========================================================================

import os, sys
import numpy as np 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .Acquisition          import Acquisition
from .BayesianNetwork      import BayesianNetwork
from .ObservationProcessor import ObservationProcessor
from .RandomSampler        import RandomSampler
from .SampleSelector       import SampleSelector
from .utilities            import ConfigParser, Logger
from .utilities            import PhoenicsNotFoundError

#========================================================================

class Phoenics(Logger):

	def __init__(self, config_file = None, config_dict = None):
	
		Logger.__init__(self, 'Phoenics', verbosity = 0)

		# parse configuration
		self.config = ConfigParser(config_file, config_dict)
		self.config.parse()
		self.config.set_home(os.path.dirname(os.path.abspath(__file__)))
	
		np.random.seed(self.config.get('random_seed'))	
		self.update_verbosity(self.config.get('verbosity'))
		self.create_folders()

		self.random_sampler       = RandomSampler(self.config.general, self.config.parameters)
		self.obs_processor        = ObservationProcessor(self.config)
		self.bayesian_network     = BayesianNetwork(self.config)
		self.acquisition          = Acquisition(self.config)
		self.sample_selector      = SampleSelector(self.config)

		self.iter_counter = 0


	def create_folders(self):

		if not os.path.isdir(self.config.get('scratch_dir')):
			try:
				os.mkdir(self.config.get('scratch_dir'))
			except FileNotFoundError:
				PhoenicsNotFoundError('Could not create scratch directory: %s' % self.config.get('scratch_dir'))

		if self.config.get_db('has_db') and not os.path.isdir(self.config.get_db('path')):
			try:
				os.mkdir(self.config.get_db('path'))
			except FileNotFoundError:
				PhoenicsNotFoundError('Could not create database directory: %s' % self.config.get_db('path'))		

		if self.config.get_db('has_db'):
			from .DatabaseHandler import DatabaseHandler
			self.db_handler = DatabaseHandler(self.config)



	def recommend(self, observations = None, as_array = False):
		
		from datetime import datetime
		start_time = datetime.now()

		if observations is None:
			# no observations, need to fall back to random sampling
			samples = self.random_sampler.draw(num = self.config.get('batches') * self.config.get('sampling_strategies'))

		elif len(observations) == 0:
			self.log('Found zero observations, falling back to random sampling', 'WARNING')
			samples = self.random_sampler.draw(num = self.config.get('batches') * self.config.get('sampling_strategies'))

		else:
			obs_params, obs_objs = self.obs_processor.process(observations)	
			
			self.bayesian_network.sample(obs_params, obs_objs)
			sampling_param_values   = self.bayesian_network.sampling_param_values
			dominant_strategy_index = self.iter_counter % len(sampling_param_values)
			dominant_strategy_value = np.array([sampling_param_values[dominant_strategy_index]])

			# prepare sample generation / selection
			best_params         = obs_params[np.argmin(obs_objs)]

			print('='*10)

			# select the remaining proposals
			proposed_samples = self.acquisition.propose(best_params=best_params, bayesian_network=self.bayesian_network, sampling_param_values=sampling_param_values, num_obs=len(obs_objs))
			samples          = self.sample_selector.select(
					self.config.get('batches'), proposed_samples, self.bayesian_network, sampling_param_values, obs_params
				)


		end_time = datetime.now()
		print('[TIME]:  ', end_time - start_time, '  (overall)')
		print('***********************************************')

		if as_array:
			# return as is
			return_samples = samples
		else:
			# convert to list of dictionaries
			param_names   = self.config.param_names
			param_types   = self.config.param_types
			sample_dicts  = []
			for sample in samples:
				sample_dict  = {}
				lower, upper = 0, self.config.param_sizes[0]
				for param_index, param_name in enumerate(param_names):
					param_type = param_types[param_index]

					sample_dict[param_name] = sample[lower:upper]

					if param_index == len(self.config.param_names) - 1:
						break
					lower  = upper
					upper += self.config.param_sizes[param_index + 1]
				sample_dicts.append(sample_dict)
			return_samples = sample_dicts

		if self.config.get_db('has_db'):
			db_entry = {'start_time': start_time, 'end_time': end_time, 
						'received_obs': observations, 'suggested_params': return_samples}
			self.db_handler.save(db_entry)

		self.iter_counter += 1
		return return_samples
		

	def read_db(self, outfile = 'database.csv', verbose = True):
		self.db_handler.read_db(outfile, verbose)


#========================================================================

if __name__ == '__main__':

	observations = [
			{'param_0': [-1.0, -1.0], 'param_1':  [1.0], 'obj_0': 0.1, 'obj_1': 0.2},
			{'param_0': [1.0, 1.0],   'param_1': [-1.0], 'obj_0': 0.2, 'obj_1': 0.1},
		]

	phoenics = Phoenics()
	samples  = phoenics.recommend(observations = observations)
