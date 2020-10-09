#!/usr/bin/env python

#==============================================================================

import copy
import time
import sqlalchemy as sql

#==============================================================================

from olympus.campaigns                 import Campaign
from olympus.databases.database_sqlite import AddEntry, FetchEntries, UpdateEntries
from olympus.utils                     import daemon, thread, generate_id

#==============================================================================

class SqliteInterface:

	SQLITE_COLUMNS = {
		'dict':   sql.PickleType(),
		'float':  sql.Float(),
		'int':    sql.Integer(),
		'pickle': sql.PickleType(),
		'string': sql.String(512),
	}

	def __init__(self, name, path, attributes, verbosity = 0):
		self.TASKS_READ   = {}
		self.TASKS_UPDATE = []
		self.TASKS_WRITE  = []

		self.attributes = attributes
		self.db_path    = f'sqlite:///{path}'
		self.name       = name

		self._create_database()
		self._process_requests()


	def _create_database(self):
		# create database
		self.db       = sql.create_engine(self.db_path)
		self.db.echo  = False
		self.metadata = sql.MetaData(self.db)

		# create table in database
		self.table = sql.Table(self.name, self.metadata)
		for attr in self.attributes:
			attr_type = 'pickle'
			if attr['type'] in self.SQLITE_COLUMNS:
				attr_type = attr['type']
			self.table.append_column(sql.Column(attr['name'].lower(), self.SQLITE_COLUMNS[attr_type]))
		self.table.create(checkfirst = True)

	#*****************************************************************************

	@thread
	def _process_requests(self):
		self.is_processing = True
		keep_processing    = True
		iter_index         = 0
		while keep_processing:
			time.sleep(0.01)
			num_tasks_read   = len(self.TASKS_READ)
			num_tasks_update = len(self.TASKS_UPDATE)
			num_tasks_write  = len(self.TASKS_WRITE)
			iter_index += 1

			# run at most one reading request
			read_keys = copy.deepcopy(list(self.TASKS_READ.keys()))
			for read_key in read_keys:
				if not self.TASKS_READ[read_key].executed:
					self.TASKS_READ[read_key].execute()
					break
			else:
				num_tasks_read = 0

			# run all update requests
			for update_index in range(num_tasks_update):
				task_update = self.TASKS_UPDATE.pop(0)
				task_update.execute()

			# run at most one writing request
			if num_tasks_write > 0:
				task_write = self.TASKS_WRITE.pop(0)
				task_write.execute()

			# clean reading reqeusts
			delete_keys = []
			for read_key in read_keys:
				if not read_key in self.TASKS_READ: continue
				if self.TASKS_READ[read_key].entries_fetched:
					delete_keys.append(read_key)
			for read_key in delete_keys:
				del self.TASKS_READ[read_key]

#			keep_processing = len(self.TASKS_READ) > 0 or len(self.TASKS_UPDATE) > 0 or len(self.TASKS_WRITE) > 0
			keep_processing = num_tasks_read > 0 or num_tasks_update > 0 or num_tasks_write > 0
		self.is_processing = False

	#*****************************************************************************

	def return_dict(function):
		def wrapper(self, *args, **kwargs):
			entries    = function(self, *args, **kwargs)
			info_dicts = [Campaign().from_dict({key['name'].lower(): entry[key['name'].lower()] for key in self.attributes}) for entry in entries]
			return info_dicts
		return wrapper

	#*****************************************************************************

	def add(self, info_dict):
		add_entry = AddEntry(self.db, self.table, info_dict)
		self.TASKS_WRITE.append(add_entry)
		if not self.is_processing:
			self._process_requests()

	@return_dict
	def fetch_all(self, cond_dict = {}):
		cond_keys = list(cond_dict)
		cond_vals = list(cond_dict)

		# define selection
		selection = sql.select([self.table])
		for index, key in enumerate(cond_keys):
			if isinstance(cond_vals[index], list):
				# with a list, we need to combine all possibilities with _or
				if len(cond_vals[index]) == 0:
					return []
				filters   = [getattr(self.table.c, key) == value for value in cond_vals[index]]
				condition = sql.or_(*filters)
			else:
				condition = getattr(self.table.c, key) == cond_vals[index]
			selection = selection.where(condition)

		# create fetch request
		fetch_entries = FetchEntries(self.db, self.table, selection, name = self.name)
		fetch_keys    = generate_id()
		self.TASKS_READ[fetch_keys] = fetch_entries
		if not self.is_processing:
			self._process_requests()

		entries = fetch_entries.get_entries()
		return entries


	def update_all(self, cond_dict, update_dict):
		cond_keys = list(cond_dict.keys())
		cond_vals = list(cond_dict.values())

		# define selection
		update = sql.update(self.table).values(update_dict).where(getattr(self.table.c, cond_keys[0]) == cond_vals[0])
		for index, key in enumerate(cond_keys[1:]):
			update = update.where(getattr(self.table.c, key) == cond_vals[index + 1])

		# submitting update
		update_entries = UpdateEntries(self.db, self.table, update)
		self.TASKS_UPDATE.append(update_entries)
		if not self.is_processing:
			self._process_requests()
