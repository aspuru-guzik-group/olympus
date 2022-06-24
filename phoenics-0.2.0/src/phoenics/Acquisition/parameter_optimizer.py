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

from utilities import Logger
from utilities import PhoenicsUnknownSettingsError

#=========================================================================

class ParameterOptimizer(Logger):

	def __init__(self, config):
		self.config = config
		Logger.__init__(self, 'ParamOptimizer', verbosity = self.config.get('verbosity'))

		# parse positions
		self.pos_continuous = np.full(self.config.num_features, False, dtype = bool)
		for feature_index, feature_type in enumerate(self.config.feature_types):
			self.pos_continuous[feature_index] = True

		# set up continuous optimization algorithms
		cont_opt_name = self.config.get('continuous_optimizer')
		if cont_opt_name == 'adam':
			from Acquisition.NumpyOptimizers import AdamOptimizer
			self.opt_con = AdamOptimizer()
		else:
			PhoenicsUnknownSettingsError('did not understand continuous optimizer "%s".\n\tPlease choose from "adam"' % cont_opt_name)


	def within_bounds(self, sample):
		return not (np.any(sample < self.config.feature_lowers) or np.any(sample > self.config.feature_uppers))


	def optimize_continuous(self, sample):
		proposal = self.opt_con.get_update(sample)
		if self.within_bounds(proposal):
			return proposal
		else:
			return sample

	
	def set_func(self, kernel, ignores = None):

		pos_continuous = self.pos_continuous.copy()
		if ignores is not None:
			for ignore_index, ignore in enumerate(ignores):
				if ignore:
					pos_continuous[ignore_index] = False

		self.opt_con.set_func(kernel, pos = np.arange(self.config.num_features)[pos_continuous])


	def optimize(self, kernel, sample, max_iter = 10):
		self.kernel = kernel

		if not self.within_bounds(sample):
			sample = np.where(sample < self.config.feature_lowers, self.config.feature_lowers, sample)
			sample = np.where(sample > self.config.feature_uppers, self.config.feature_uppers, sample)
			sample = sample.astype(np.float32)

		# update all optimization algorithms
		sample_copy = sample.copy()
		optimized   = sample.copy()
		for num_iter in range(max_iter):

			# one step of adam
			if np.any(self.pos_continuous):
				optimized = self.optimize_continuous(optimized)

			# check for convergence
			if np.any(self.pos_continuous) and np.linalg.norm(sample_copy - optimized) < 1e-7:
				break
			else:
				sample_copy = optimized.copy()

		return optimized
	
