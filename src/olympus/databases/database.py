#!/usr/bin/env python

# ===============================================================================

from olympus import Logger
from olympus.utils import generate_id

# ===============================================================================


class Database:
    """ generic database collector - can connect to multiple databases

    This class is intended to provide the interface to other modules within
    olympus.
    """

    def __init__(self, kind="sqlite", *args, **kwargs):
        self.dbs = {}
        self.active_db = None
        if not kind is None:
            self.add_db(kind, *args, **kwargs)

    def __iter__(self):
        for db in self.dbs.values():
            for campaign in db.get_campaigns():
                yield campaign

    def __repr__(self):
        return self.db.__repr__()

    @property
    def file_types(self):
        file_types = []
        for db in self.dbs.values():
            file_types.extend(db.file_types)
        return file_types

    @property
    def is_processing(self):
        return self.active_db.is_processing

    @property
    def db(self):
        return self.active_db

    def is_valid_file_type(self, file_type):
        return self.db._validate(file_type=file_type)

    def _guess_db_kind(self, file_name):
        from . import databases

        file_type = file_name.split(".")[-1]
        for db_kind, db in databases.items():
            if db.is_valid_file_type(file_type):
                break
        else:
            from . import db_types

            Logger.log(
                "Could not find database type {}. Please choose from {}".format(
                    file_type, db_types
                ),
                "ERROR",
            )
            return None
        return db_kind

    def from_file(self, file_name, kind=None):
        """ connects to a database stored on disk

        Args:
            file_name (str): path and name of the database file

        Returns:
            (Database): loaded database
        """
        if kind is None:
            kind = self._guess_db_kind(file_name)
        if kind is None:
            Logger.log(
                "Please provide database format (automatic guessing failed)", "ERROR"
            )
            return None
        db_name = file_name.split("/")[-1].split(".")[0]
        if db_name in self.dbs:
            self.set_active_db(db_name)
        else:
            self.add_db(kind=kind, name=db_name)
            self.set_active_db(db_name)
        self.db.from_file(file_name)
        return self

    def set_active_db(self, db_name):
        self.active_db = self.dbs[db_name]

    def add_db(self, kind, *args, **kwargs):
        try:
            database = __import__(
                f"olympus.databases.database_{kind}", fromlist=[f"Wrapper_{kind}"]
            )
        except ModuleNotFoundError:
            Logger.log(" ... proceeding with pickle database", "INFO", only_once=True)
            kind = "pickle"
            database = __import__(
                f"olympus.databases.database_{kind}", fromlist=[f"Wrapper_{kind}"]
            )

        database = getattr(database, f"Wrapper_{kind}")
        db = database(*args, **kwargs)
        self.dbs[db.name] = db
        if self.active_db is None:
            self.active_db = db

    def get_campaigns(self):
        campaigns = []
        for db in self.dbs.values():
            campaigns.extend(db.get_campaigns())
        return campaigns

    def list_campaigns(self):
        campaign_ids = []
        for db in self.dbs.values():
            campaign_ids.extend(db.list_campaigns())
        return campaign_ids

    def log_campaign(self, campaign):
        self.active_db.log_campaign(campaign)


# ===============================================================================
