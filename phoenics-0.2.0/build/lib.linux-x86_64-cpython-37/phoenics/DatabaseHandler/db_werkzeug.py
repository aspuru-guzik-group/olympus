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

from .           import DB_Cache
from ..utilities import Logger
from ..utilities import PhoenicsUnknownSettingsError

#========================================================================

class DB_Werkzeug(Logger):


	def __init__(self, config, db_attributes, verbosity = 0):
		Logger.__init__(self, 'DB_Werkzeug', verbosity = verbosity)
		self.config   = config      
		self.db_attrs = db_attributes


	def create_database(self):
		if not self.config.get_db('data_storage'):
			self.log('No data storage specified. Phoenics will not store any information.', 'INFO')
		elif self.config.get_db('data_storage') == 'sqlite':
			from .SqliteInterface import SqliteDatabase
			self.database = SqliteDatabase(self.config.get_db('path'), self.db_attrs, 'db', verbosity = self.config.get('verbosity'))
		else:
			PhoenicsUnknownSettingsError('did not understand data storage: "%s".\n\tChoose from [null, "sqlite"]' % self.config.get_db('data_storage')) 


	def create_cache(self):
		all_entries = self._get({})
		self.cache  = DB_Cache(self.db_attrs, all_entries, verbosity = self.config.get('verbosity'))


	def _get(self, condition):
		entries = self.db_fetch_all(condition)
		return entries


	#=== CHECK EXISTANCE ===#
	
	def db_check_existance(self, condition):
		if hasattr(self, 'cache'):
			return self._check_existance_cache(condition)
		return self._check_existance_database(condition)


	def _check_existance_cache(self, condition):
		for attr, value in condition.items():
			if not value in self.cache[attr]:
				return False
		return True


	def _check_existance_database(self, condition):
		entries = self._get(condition)
		return len(entries) > 0


	#=== ADD TO DB ===#
	
	def db_add(self, info_dict):
		if hasattr(self, 'cache'):
			self._add_cache(info_dict)
		self._add_database(info_dict)

	def _add_cache(self, info_dict):
		self.cache.add(info_dict)

	def _add_database(self, info_dict):
		try:
			self.database.add(info_dict)
		except AttributeError:
			info_dict_str = ''
			for key, item in info_dict.items():
				info_dict_str = '%s:\t%s\n' % (str(key), str(item))
			self.log('could not add to database:\n%s' % (info_dict_str), 'ERROR')


	#=== FETCH FROM DB ===#

	def db_fetch_all(self, condition_dict = {}):

		if hasattr(self, 'cache'):
			entries = self._fetch_all_cache(condition_dict)
		else:
			entries = self._fetch_all_database(condition_dict)
		return entries


	def _fetch_all_cache(self, condition_dict):
		return self.cache.fetch_all(condition_dict)

	def _fetch_all_database(self, condition_dict):
		try:
			return self.database.fetch_all(condition_dict)
		except OSError:
			condition_dict_str = ''
			for key, item in condition_dict.items():
				condition_dict_str = '%s:\t%s\n' % (str(key), str(item))
			self.log('could not fetch from database:\n%s' % (condition_dict_str), 'ERROR')

	
	#=== UPDATE DB ===#

	def db_update_all(self, condition_dict, update_dict):
		if hasattr(self, 'cache'):
			self._update_all_cache(condition_dict, update_dict)
		self._update_all_database(condition_dict, update_dict)

	def _update_all_cache(self, condition_dict, update_dict):
		self.cache.update_all(condition_dict, update_dict)

	def _update_all_database(self, condition_dict, update_dict):
		try:
			self.database.update_all(condition_dict, update_dict)
		except OSError:
			condition_dict_str = ''
			for key, item in condition_dict.items():
				condition_dict_str = '%s:\t%s\n' % (str(key), str(item))
			self.log('could not find entry with conditions in database:\n%s' % (condition_dict_str), 'ERROR')

















