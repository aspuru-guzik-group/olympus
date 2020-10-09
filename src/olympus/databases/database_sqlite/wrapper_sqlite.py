#!/usr/bin/env python

#==============================================================================

import sqlalchemy

#==============================================================================

from olympus.campaigns import Campaign
from olympus.databases import AbstractDatabase
from olympus.databases.database_sqlite import SqliteInterface

#==============================================================================

class Wrapper_sqlite(AbstractDatabase):

	''' implements a simple sqlite database

	Architecture of the database:
		image of the `Campaign` class, where entries encode individual objects
	'''

	ATT_KIND      = {'type': 'string', 'default': 'sqlite'}
	ATT_FILE_TYPE = {'type': 'string', 'default': 'sqlite', 'valid': ['db', 'sqlite']}

	def __init__(self, *args, **kwargs):
		AbstractDatabase.__init__(self, *args, **kwargs)
		self.logged_campaign_ids = []

	def __getattr__(self, prop):
		if prop == 'db':
			self.db = SqliteInterface(self.name, self.file_name, self.db_attrs())
			return self.db
		else:
			return AbstractDatabase.__getattr__(self, prop)


	def db_attrs(self):
		db_attrs = Campaign().defaults
		return db_attrs

	def _get_campaigns(self):
		campaigns = self.db.fetch_all()
		return campaigns

	def _list_campaign_ids(self):
		campaigns = self.db.fetch_all()
		return [campaign.id for campaign in campaigns]

	def _load_db(self):
		return self.db.fetch_all()

	def _log_campaign(self, campaign):
		if campaign.id in self.logged_campaign_ids:
			self.db.update_all({'id': campaign.id}, campaign.to_dict())
		else:
			self.db.add(campaign.to_dict())
			self.logged_campaign_ids.append(campaign.id)
