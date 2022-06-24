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

from ObservationProcessor import Chimera
from utilities import Logger
from utilities import PhoenicsUnknownSettingsError

#=======================================================================

class ObservationProcessor(Logger):

	def __init__(self, config):
		self.config  = config
		self.chimera = Chimera(self.config.obj_tolerances, self.config.get('softness'))
		Logger.__init__(self, 'ObservationProcessor', verbosity = self.config.get('verbosity'))
		
		# compute some boundaries
		self.feature_lowers = self.config.feature_lowers
		self.feature_uppers = self.config.feature_uppers
		self.soft_lower     = self.feature_lowers + 0.1 * (self.feature_uppers - self.feature_lowers)
		self.soft_upper     = self.feature_uppers - 0.1 * (self.feature_uppers - self.feature_lowers)


	def adjust_objectives(self, objs):
		'''adjust objectives based on optimization goal'''
		optim_goals   = self.config.obj_goals	
		adjusted_objs = np.empty(objs.shape)
		for obj_index, obj_goal in enumerate(optim_goals):
			if obj_goal == 'minimize':
				adjusted_objs[:, obj_index] =   objs[:, obj_index]
			elif obj_goal == 'maximize':
				adjusted_objs[:, obj_index] = - objs[:, obj_index]
			else:
				PhoenicsUnknownSettingsError('did not understand objective goal: "%s" for objective "%s".\n\tChoose from "minimize" or "maximize"' % (obj_goal, self.config.obj_names[obj_index]))
		return adjusted_objs


	def mirror_parameters(self, param_vector):
		# get indices
		lower_indices_prelim = np.where(param_vector < self.soft_lower)[0]	
		upper_indices_prelim = np.where(param_vector > self.soft_upper)[0]
	
		lower_indices, upper_indices = [], []
		for feature_index, feature_type in enumerate(self.config.feature_types):
			if feature_index in lower_indices_prelim:
				lower_indices.append(feature_index)
			if feature_index in upper_indices_prelim:
				upper_indices.append(feature_index)

		index_dict    = {index: 'lower' for index in lower_indices}
		for index in upper_indices:
			index_dict[index] = 'upper'

		# mirror param
		params = []
		index_dict_keys, index_dict_values = list(index_dict.keys()), list(index_dict.values())
		for index in range(2**len(index_dict)):
			param_copy = param_vector.copy()
			for jndex in range(len(index_dict)):
				if (index // 2**jndex) % 2 == 1:
					param_index = index_dict_keys[jndex]
					if index_dict_values[jndex] == 'lower':
						param_copy[param_index] = self.feature_lowers[param_index] - (param_vector[param_index] - self.feature_lowers[param_index])
					elif index_dict_values[jndex] == 'upper':
						param_copy[param_index] = self.feature_uppers[param_index] + (self.feature_uppers[param_index] - param_vector[param_index])
			params.append(param_copy)
		if len(params) == 0:
			params.append(param_vector.copy())
		return params


	def scalarize_objectives(self, objs):
		scalarized = self.chimera.scalarize(objs)
		min_obj, max_obj = np.amin(scalarized), np.amax(scalarized)
		if min_obj != max_obj:
			scaled_obj = (scalarized - min_obj) / (max_obj - min_obj)
			scaled_obj = np.sqrt(scaled_obj)
		else:
			scaled_obj = scalarized - min_obj
		return scaled_obj



	def process(self, obs_dicts):

		param_names   = self.config.param_names
		param_types   = self.config.param_types

		# get raw results
		raw_params, raw_objs = [], []
		for obs_dict in obs_dicts:
			
			# get param-vector
			param_vector = []
			for param_index, param_name in enumerate(param_names):
				obs_param = obs_dict[param_name]
				param_vector.extend(obs_param)
	
			mirrored_params = self.mirror_parameters(param_vector)

			# get obj-vector
			obj_vector = np.array([obs_dict[obj_name] for obj_name in self.config.obj_names])

			# add processed params
			for param in mirrored_params:
				raw_params.append(param)
				raw_objs.append(obj_vector)

		raw_objs, raw_params = np.array(raw_objs), np.array(raw_params)
		params = raw_params
	
		adjusted_objs = self.adjust_objectives(raw_objs)
		objs          = self.scalarize_objectives(adjusted_objs)

		return params, objs


