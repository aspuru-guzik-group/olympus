#!/usr/bin/env python

#===============================================================================

import pickle

#===============================================================================

from olympus.databases import AbstractDatabase

#===============================================================================

class Wrapper_pickle(AbstractDatabase):

    ''' implements a simple database using pickle

    Architecture of the database:
        giant dictionary where keys are the identifiers of individual campaigns
        and the associated attributes represent the campaign as yet another
        dictionary
    '''

    ATT_KIND      = {'type': 'string', 'default': 'pkl'}
    ATT_FILE_TYPE = {'type': 'string', 'default': 'pkl', 'valid': ['pkl', 'pickle']}


    def _load_db(self):
        # ATTENTION: requires a lock if we ever run into race conditions
        with open(self.file_name, 'rb') as content:
            db = pickle.load(content)
        self.db = db
        return self


    def _list_campaign_ids(self):
        self._load_db()
        return list(self.db.keys())


    def _log_campaign(self, campaign):
        if self.db_exists:
            self._load_db()
        else:
            self.db = {}
        self.db[campaign.id] = campaign.to_dict()
        self._save_db(self.db)


    def _save_db(self, db):
        with open(self.file_name, 'wb') as content:
            pickle.dump(db, content)





#===============================================================================
