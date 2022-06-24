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
import sys
import json
import numpy as np 

from utilities import PhoenicsParseError, PhoenicsValueError
from utilities import Logger

from utilities import ParserJSON #, CategoryParser
from utilities import default_general_configurations
from utilities import default_database_configurations
from utilities import safe_execute

#=========================================================================

class Configuration(object):

	def __init__(self, me = ''):
		self.me          = me + ':'
		self.added_props = []
		self.added_attrs = []

	def __str__(self):
		new_line = '%s\n' % self.me
		for prop in sorted(self.added_props):
			new_line += '--> %s:\t%s\n' % (prop, getattr(self, prop))
		return(new_line)


	def __iter__(self):
		for _ in range(self.num_elements):
			info_dict = {}
			for prop_index, prop in enumerate(self.added_props):
				info_dict[prop] = self.added_attrs[prop_index][_]
			yield info_dict


	def __getitem__(self, index):
		info_dict = {}
		for prop_index, prop in enumerate(self.added_props):
			info_dict[prop] = self.added_attrs[prop_index][index]
		return info_dict

	
	def to_dict(self):
		return {prop: getattr(self, prop) for prop in sorted(self.added_props)}


	def add_attr(self, prop, attr):
		setattr(self, prop, attr)
		if not prop in self.added_props:		
			self.added_props.append(prop)
			self.added_attrs.append(attr)
		try:
			self.num_elements = len(attr)
		except TypeError:
			pass

	def get_attr(self, prop):
		return getattr(self, prop)

#========================================================================

class ConfigParser(Logger):

	TYPE_ATTRIBUTES = {
			'continuous':  ['low', 'high'],
		}


	def __init__(self, config_file = None, config_dict = None):

		Logger.__init__(self, 'ConfigParser', verbosity = 0)
		self.config_file     = config_file
		self.config_dict     = config_dict

	def _parse_general(self, provided_settings):
		self.general = Configuration('general')
		for general_key, general_value in default_general_configurations.items():
			if general_key in provided_settings:
				general_value = provided_settings[general_key]
			if general_value in ['True', 'False']:
				general_value = general_value == 'True'
			self.general.add_attr(general_key, general_value)


	def _parse_database(self, provided_settings):
		self.database = Configuration('database')
		for general_key, general_value in default_database_configurations.items():
			if general_key in provided_settings:
				general_value = provided_settings[general_key]
			if general_value in ['True', 'False']:
				general_value = general_value == 'True'
			self.database.add_attr(general_key, general_value)
		if self.database.data_storage in ['sqlite']:
			self.database.add_attr('has_db', True)
		else:
			self.database.add_attr('has_db', False)


	def _parse_parameters(self, provided_settings):
		self.parameters   = Configuration('parameters')
		self.features     = Configuration('features')
		self.kernels      = Configuration('kernels')
		param_configs     = {'name': [], 'type': [], 'size': [], 'specifics': [], 'process_constrained': []}
		feature_configs   = {'name': [], 'type': [], 'size': [], 'specifics': [], 'process_constrained': []}
		kernel_configs    = {'name': [], 'type': [], 'size': [], 'specifics': [], 'process_constrained': []}

		if len(provided_settings) == 0:
			self.log('need to define at least one parameter', 'FATAL')
		
		# parse parameter configuration
		for setting in provided_settings:

			size     = setting['size']
			num_cats = 1

			if 'process_constrained' in setting:
				setting['process_constrained'] = bool(setting['process_constrained'])
			else:
				setting['process_constrained'] = False
				

			if setting['type'] == 'continuous':
				# check order
				if setting['high'] <= setting['low']:
					PhoenicsValueError('upper limit (%f) needs to be larger than the lower limit (%f) for parameter "%s"' % (setting['high'], setting['low'], setting['name']))
			else:
				self.log('Did not understand parameter type "%s" for parameter "%s". Please choose from "continuous".' % (setting['type'], setting['name']), 'FATAL')


			for key in param_configs.keys():
				if key == 'specifics':
					element = {spec_key: setting[spec_key] for spec_key in self.TYPE_ATTRIBUTES[setting['type']]}
				else:
					element = setting[key]

				param_configs[key].append(element)
				feature_configs[key].extend([element for _ in range(size)])
				kernel_configs[key].extend([element for _ in range(size * num_cats)])


		# write configuration
		for key in param_configs.keys():
			self.parameters.add_attr(key, param_configs[key])
			self.features.add_attr(key, feature_configs[key])
			self.kernels.add_attr(key, kernel_configs[key])


	def _parse_objectives(self, provided_settings):
		self.objectives = Configuration('objectives')
		obj_configs     = {'name': [], 'goal': [], 'hierarchy': [], 'tolerance': []}

		if len(provided_settings) == 0:
			self.log('need to define at least one objective', 'FATAL')

		elif len(provided_settings) == 1:
			setting = provided_settings[0]
			obj_configs['name'].append(setting['name'])
			obj_configs['goal'].append(setting['goal'])
			obj_configs['hierarchy'].append(0)
			obj_configs['tolerance'].append(0.0)

		else:	
			for setting in provided_settings:
				for key in obj_configs.keys():
					obj_configs[key].append(setting[key])

		# sort entries based on hierachy
		sort_indices = np.argsort(obj_configs['hierarchy'])		
		
		# check that hierarchies are complete
		for hier_index in range(len(sort_indices)):
			if np.abs(hier_index - obj_configs['hierarchy'][sort_indices[hier_index]]) > 1e-4:
				self.log('incorrect objective hierarchy\n\t... please sort objective hierarchies in ascending order, starting from 0', 'FATAL')
		
		# write configuration
		for key in obj_configs:
			self.objectives.add_attr(key, np.array(obj_configs[key])[sort_indices])

	@property
	def settings(self):
		settings_dict = {}
		settings_dict['general']    = self.general.to_dict()
		settings_dict['database']   = self.database.to_dict()
		settings_dict['parameters'] = self.parameters.to_dict()
		settings_dict['objectives'] = self.objectives.to_dict()
		return settings_dict

	@property
	def process_constrained(self):
		is_constrained = np.any(self.parameters.process_constrained)
		return is_constrained

	@property
	def param_names(self):
		return self.parameters.name

	@property
	def param_types(self):
		return self.parameters.type

	@property
	def param_sizes(self):
		return self.parameters.size

	@property
	def feature_process_constrained(self):
		return self.features.process_constrained


	@property
	def feature_lengths(self):
		lengths = []
		for spec in self.features.specifics:
			lengths.append(spec['high'] - spec['low'])
		return np.array(lengths)

	@property
	def feature_lowers(self):
		lowers = []
		for spec in self.features.specifics:
			lowers.append(spec['low'])
		return np.array(lowers)

	@property
	def feature_names(self):
		return self.features.name

	@property
	def feature_ranges(self):
		return self.feature_uppers - self.feature_lowers

	@property
	def feature_sizes(self):
		sizes = []
		for feature_index, feature_type in enumerate(self.feature_types):
			feature_size = 1
			sizes.append(feature_size)
		return np.array(sizes)

	@property
	def feature_types(self):
		return self.features.type

	@property
	def feature_uppers(self):
		uppers = []
		for spec in self.features.specifics:
			if 'high' in spec:
				uppers.append(spec['high'])
			else:
				uppers.append(1.)
		return np.array(uppers)

	@property
	def num_features(self):
		return len(self.feature_names)


	@property
	def kernel_lowers(self):
		lowers = []
		for spec in self.kernels.specifics:
			lowers.append(spec['low'])
		return np.array(lowers)

	@property
	def kernel_names(self):
		return self.kernels.name

	@property
	def kernel_ranges(self):
		return self.kernel_uppers - self.kernel_lowers

	@property
	def kernel_sizes(self):
		sizes = []
		for kernel_index, kernel_type in enumerate(self.kernel_types):
			kernel_size = 1
			sizes.append(kernel_size)
		return np.array(sizes)

	@property
	def kernel_types(self):
		return self.kernels.type

	@property
	def kernel_uppers(self):
		uppers = []
		for spec in self.kernels.specifics:
			uppers.append(spec['high'])
		return np.array(uppers)


	@property
	def obj_names(self):
		return self.objectives.name

	@property
	def obj_tolerances(self):
		return self.objectives.tolerance
		
	@property
	def obj_goals(self):
		return self.objectives.goal

	def get(self, attr):
		return self.general.get_attr(attr)

	def get_db(self, attr):
		return self.database.get_attr(attr)

	def set_home(self, home_path):
		self.general.add_attr('home', home_path)


	def _parse(self, config_dict):
		self.config = config_dict
		
		if 'general' in self.config:
			self._parse_general(self.config['general'])
		else:
			self._parse_general({})
		self.update_verbosity(self.general.verbosity)

		if 'database' in self.config:
			self._parse_database(self.config['database'])
		else:
			self._parse_database({})

		self._parse_parameters(self.config['parameters'])
		self._parse_objectives(self.config['objectives'])


	@safe_execute(PhoenicsParseError)
	def parse_config_file(self, config_file = None):

		if not config_file is None:
			self.config_file = config_file

		self.json_parser = ParserJSON(json_file = self.config_file)
		self.config_dict = self.json_parser.parse()
		self._parse(self.config_dict)


	@safe_execute(PhoenicsParseError)
	def parse_config_dict(self, config_dict = None):
	
		if not config_dict is None:
			self.config_dict = config_dict

		self._parse(self.config_dict)


	def parse(self):
		# test if both dict and file have been provided
		if self.config_dict is not None and self.config_file is not None:
			self.log('Found both configuration file and configuration dictionary. Will parse configuration from dictionary and ignore file', 'WARNING')
			self.parse_config_dict(self.config_dict)
		elif self.config_dict is not None:
			self.parse_config_dict(self.config_dict)
		elif self.config_file is not None:
			self.parse_config_file(self.config_file)
		else:
			self.log('Cannot parse configuration due to missing configuration file or configuration dictionary', 'ERROR')



	
