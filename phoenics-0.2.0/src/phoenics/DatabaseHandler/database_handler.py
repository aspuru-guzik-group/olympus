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

import numpy as np 
from   datetime import datetime

from .           import DB_Werkzeug
from ..utilities import Logger
from ..utilities import PhoenicsUnknownSettingsError, PhoenicsValueError

#========================================================================

class DatabaseHandler(DB_Werkzeug, Logger):

	DB_ATTRIBUTES = {
			'end_time':           'string',
			'received_obs':       'pickle', 
			'runtime':            'string',
			'start_time':         'string',
			'suggested_params':   'pickle',
		}


	def __init__(self, config):
		self.config = config
		DB_Werkzeug.__init__(self, self.config, self.DB_ATTRIBUTES, verbosity = self.config.get('verbosity'))
		Logger.__init__(self, 'DatabaseHandler', self.config.get('verbosity'))

		self.create_database()
		self.create_cache()


	def save(self, db_entry):
		if self.config.get_db('log_runtimes'):
			db_entry['runtime'] = db_entry['end_time'] - db_entry['start_time']
			for attr in ['start_time', 'end_time', 'runtime']:
				db_entry[attr] = str(db_entry[attr])
		else:
			for attr in ['start_time', 'end_time', 'runtime']:
				db_entry[attr] = 'n/a'
		if not self.config.get_db('log_observations'):
			db_entry['received_obs'] = []
		self.db_add(db_entry)


	def read_db(self, outfile, verbose):
	
		db_content = self.db_fetch_all()
		if len(db_content) == 0:
			PhoenicsValueError('no entries found in database')
		out_format = outfile.split('.')[-1]
		if not out_format in ['csv', 'xlsx', 'pkl', 'json']:
			PhoenicsUnknownSettingsError('did not understand output format "%s".\n\tPlease choose from "csv", "json", "pkl" or "xlsx"' % out_format)

		# sort entries
		if self.config.get_db('log_runtimes'):
			start_times     = [datetime.strptime(entry['start_time'], '%Y-%m-%d %H:%M:%S.%f') for entry in db_content]
			sorting_indices = np.argsort(start_times)
		else:
			sorting_indices = np.arange(len(db_content))

		# create output dictionary
		relevant_keys         = ['start_time', 'end_time', 'runtime']
		first_suggested_batch = db_content[0]['suggested_params']
		len_batch             = len(first_suggested_batch)
		param_names           = list(first_suggested_batch[0].keys())
		for sugg_index in range(len_batch):
			for param_name in param_names:
				relevant_keys.append('%s (%d)' % (param_name, sugg_index))
		db_dict = {key: [] for key in relevant_keys}

		for sorting_index in sorting_indices:
			entry = db_content[sorting_index]
			for key in entry.keys():
				if key == 'suggested_params':
					for sugg_index in range(len_batch):
						for param_name in param_names:
							if not param_name in entry[key][sugg_index]:
								PhoenicsValueError('could not find parameter "%s" in db entry' % param_name)
							sugg_params = np.squeeze(entry[key][sugg_index][param_name])
							db_key      = '%s (%d)' % (param_name, sugg_index)
							db_dict[db_key].append(sugg_params)
				else:
					if not key in relevant_keys: continue
					db_dict[key].append(entry[key])

		# set up writer
		if out_format in ['csv', 'xlsx']:
			from DatabaseHandler.PandasWriters import DB_Writer
		elif out_format in ['json']:
			from DatabaseHandler.JsonWriters   import DB_Writer
			db_dict['config'] = self.config.settings
		elif out_format in ['pkl']:
			from DatabaseHandler.PickleWriters import DB_Writer

		self.db_writer = DB_Writer(self.config)
		self.db_writer.write(db_dict, outfile, out_format)


