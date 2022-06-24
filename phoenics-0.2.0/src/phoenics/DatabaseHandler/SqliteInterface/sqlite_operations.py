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

import time 
import sqlalchemy as sql

#========================================================================

class AddEntry(object):

	def __init__(self, database, table, entry):
		self.db    = database
		self.table = table 
		self.entry = entry

	def execute(self):
		with self.db.connect() as conn:
			conn.execute(self.table.insert(), self.entry)
			conn.close()

#========================================================================

class FetchEntries(object):
	
	def __init__(self, database, table, selection):
		self.db              = database
		self.table           = table
		self.selection       = selection
		self.entries         = None
		self.executed        = False
		self.entries_fetched = False

	def execute(self):
		with self.db.connect() as conn:
			selected  = conn.execute(self.selection)
			entries   = selected.fetchall()
			conn.close()
		self.entries  = entries
		self.executed = True

	def get_entries(self):
		iteration_index = 0
		while not self.executed:
			time.sleep(0.02)
		self.entries_fetched = True
		return self.entries

#========================================================================

class UpdateEntries(object):

	def __init__(self, database, table, updates):
		self.db      = database
		self.table   = table
		self.updates = updates


	def execute(self):
		with self.db.connect() as conn:
			updated = conn.execute(self.updates)
			conn.close()










