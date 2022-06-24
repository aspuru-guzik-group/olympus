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

import numpy  as np 
import pandas as pd
from datetime import datetime

from utilities import Logger

#=======================================================================

class Writer(object):
	def __init__(self, file_name):
		self.file_name = file_name

	def save(self):
		pass

class CsvWriter(Writer):
	
	def __init__(self, file_name):
		Writer.__init__(self, file_name)

	def __call__(self, pd_frame):
		pd_frame.to_csv(self.file_name)


class ExcelWriter(Writer):

	def __init__(self, file_name):
		Writer.__init__(self, file_name)
		self.writer = pd.ExcelWriter(self.file_name)

	def __call__(self, pd_frame):
		pd_frame.to_excel(self.writer, 'Sheet1')

	def save(self):
		self.writer.save()

#=======================================================================


class DB_Writer(Logger):

	def __init__(self, config):
		self.config = config
		Logger.__init__(self, 'DB_Writer', self.config.get('verbosity'))


	def create_writer(self, file_name, out_format):
		if out_format == 'xlsx':
			self.writer = ExcelWriter(file_name)
		elif out_format == 'csv':
			self.writer = CsvWriter(file_name)


	def write(self, db_content, outfile, out_format):

		# create the writer
		self.create_writer(outfile, out_format)

		# sort entries
#		if self.config.get_db('log_runtimes'):
#			start_times     = [datetime.strptime(entry['start_time'], '%Y-%m-%d %H:%M:%S.%f') for entry in db_content]
#			sorting_indices = np.argsort(start_times)
#		else:
#			sorting_indices = np.arange(len(db_content))
#
#
#		# create output dictionary
#		relevant_keys         = ['start_time', 'end_time', 'runtime']
#		first_suggested_batch = db_content[0]['suggested_params']
#		len_batch             = len(first_suggested_batch)
#		param_names           = list(first_suggested_batch[0].keys())
#		for sugg_index in range(len_batch):
#			for param_name in param_names:
#				relevant_keys.append('%s (%d)' % (param_name, sugg_index))
#		db_dict = {key: [] for key in relevant_keys}
#
#		for sorting_index in sorting_indices:
#			entry = db_content[sorting_index]
#			for key in entry.keys():
#				if key == 'suggested_params':
#					for sugg_index in range(len_batch):
#						for param_name in param_names:
#							sugg_params = np.squeeze(entry[key][sugg_index][param_name])
#							db_key      = '%s (%d)' % (param_name, sugg_index)
#							db_dict[db_key].append(sugg_params)
#				else:	
#					if not key in relevant_keys: continue
#					db_dict[key].append(entry[key])				

		# convert output dict and save via pandas routine
		dataframe = pd.DataFrame.from_dict(db_content)
		self.writer(dataframe)
		self.writer.save()

	


