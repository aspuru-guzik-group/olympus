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

from utilities import Logger
from utilities import PhoenicsUnknownSettingsError

#========================================================================

class RandomSampler(Logger):

	def __init__(self, config_general, config_params):
		self.config_general = config_general
		self.config_params  = config_params
		verbosity           = self.config_general.verbosity
		if 'random_sampler' in self.config_general.verbosity:
			verbosity = self.config_general.verbosity['random_sampler']
		Logger.__init__(self, 'RandomSampler', verbosity)

		if self.config_general.sampler == 'sobol':
			from RandomSampler.sobol   import SobolContinuous
			self.continuous_sampler  = SobolContinuous()
		elif self.config_general.sampler == 'uniform':
			from RandomSampler.uniform import UniformContinuous
			self.continuous_sampler  = UniformContinuous()
		else:
			PhoenicsUnknownSettingsError('did not understanding sampler setting: "%s".\n\tChoose from "uniform" or "sobol"' % self.config_general.sampler)


	def draw(self, num = 1):
		samples = []
		for param_index, param_settings in enumerate(self.config_params):
			specs = param_settings['specifics']
			sampled_values = self.continuous_sampler.draw(specs['low'], specs['high'], (num, param_settings['size']))
			samples.append(sampled_values)
		samples = np.concatenate(samples, axis = 1)
		self.log('generated uniform samples: \n%s' % str(samples), 'DEBUG')
		return samples


	def perturb(self, pos, num = 1, scale = 0.05):
		samples = []
		for param_index, param_settings in enumerate(self.config_params):
			specs = param_settings['specifics']
			sampled_values  = self.continuous_sampler.draw(-scale, scale, (num, param_settings['size']))
			sampled_values *= specs['high'] - specs['low']
			close_samples   = pos[param_index] + sampled_values
			close_samples   = np.where(close_samples < specs['low'],  specs['low'],  close_samples)
			close_samples   = np.where(close_samples > specs['high'], specs['high'], close_samples)
			samples.append(close_samples)
		samples = np.concatenate(samples, axis = 1)
		return samples


	def normal_samples(self, loc = 0., scale = 1., num = 1):
		samples = []
		for param_index, param_settings in enumerate(self.config_params):
			specs = param_settings['specifics']
			param_range = specs['high'] - specs['low']
			sampled_values = np.random.normal(0., scale * param_range, (num, param_settings['size'])) + loc[param_index]
			samples.append(sampled_values)
		samples = np.concatenate(samples, axis = 1)
		return samples				

