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

The code in this file was developed at ChemOS Inc. (2019).
'''

__author__  = 'Florian Hase'

#=========================================================================

import copy
import json
import numpy as np 

from utilities import Logger

#=======================================================================

class DB_Writer(Logger):

	def __init__(self, config):
		self.config = config
		Logger.__init__(self, 'DB_Writer', self.config.get('verbosity'))


	def write(self, db_content, outfile, out_format):

		db_clean = {}
		for key in db_content:

			if key in ['config']: 

				db_clean[key] = {}
				db_clean[key]['general']    = db_content[key]['general']
				db_clean[key]['database']   = db_content[key]['database']

				# list-ify parameters
				db_clean[key]['parameters'] = {}
				for dict_key in db_content[key]['parameters']:
					if dict_key == 'specifics': 
						db_clean[key]['parameters'][dict_key] = []
						for element in db_content[key]['parameters'][dict_key]:
							if 'category_details' in element:
								db_clean[key]['parameters'][dict_key].append({'category_details': element['category_details']})
						continue
					db_clean[key]['parameters'][dict_key] = db_content[key]['parameters'][dict_key]

				# list-ify objectives
				db_clean[key]['objectives'] = {dict_key: db_content[key]['objectives'][dict_key].tolist() for dict_key in db_content[key]['objectives']}
				continue

#			if key in ['descriptor_summary']:
#				
#				summary       = db_content[key][-1]
#				clean_summary = {}
#
#				name       = '-1'
#				name_index = 0
#				for feature_index, feature_name in enumerate(self.config.feature_names):
#					if feature_name != name:
#						name = feature_name
#						name_index = 0
#
#					sum_dict = summary['feature_%d' % feature_index]
#					for key, value in sum_dict.items():
#						sum_dict[key] = {inner_key: value[inner_key].tolist() for inner_key in value.keys()}
#	
#					key_name = '%s (%d)' % (name, name_index)
#					clean_summary[key_name] = copy.deepcopy(sum_dict)
#					name_index += 1
#	
#				db_clean['descriptor_summary'] = clean_summary
#				continue

			values = db_content[key]
			if isinstance(values, list):
				first_element = values[0]
				if isinstance(first_element, np.ndarray):
					new_values = []
					for element in values:
						if len(element.shape) == 0:
							new_values.append(element.item())
						else:
							new_values.append([atom_element.item() for atom_element in element])				
				else:
					new_values = values
			else: 
				continue
			db_clean[key] = new_values


		formatted_string = json.dumps(db_clean, indent = 4, separators = (',', ': '))
		content = open(outfile, 'w')
		content.write(formatted_string)
		content.close()

