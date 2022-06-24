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

from utilities import Logger

#=========================================================================

class DB_Cache(Logger):

	def __init__(self, attributes, entries = [], verbosity = 0):
		Logger.__init__(self, 'DB_Cache', verbosity = verbosity)
		self.attributes = attributes		

		self.cache     = {attr: [] for attr in self.attributes}
		self.num_items = 0
		for entry in entries:
			self.add(entry)


	def __getitem__(self, item):
		try:
			return self.cache[item]
		except KeyError:
			return []



	def add(self, info_dict):
		for attr in self.attributes:
			if attr in info_dict:
				self.cache[attr].append(info_dict[attr])
			else:	
				self.cache[attr].append(None)
		self.num_items += 1


	def fetch_all(self, condition_dict):
		results = []
		for element_index in range(self.num_items):
			for key, value in condition_dict.items():
				if value != self.cache[key][element_index]:
					break
			else:
				result = {attr: self.cache[attr][element_index] for attr in self.attributes}
				results.append(result)
		return results


	def update_all(self, condition_dict, update_dict):
		for element_index in range(self.num_items):
			for key, value in condition_dict.items():
				if value != self.cache[key][element_index]:
					break
			else:
				for key, value in update_dict.items():
					self.cache[key][element_index] = value


