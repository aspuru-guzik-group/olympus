#!/usr/bin/env python

# ===============================================================================

import os

from olympus import __scratch__
from olympus import Logger
from olympus.objects import Object
from olympus.utils import generate_id

# ===============================================================================


class AbstractDatabase(Object):

    """ bridge to specific databases; implements in-memory cache
    """

    ATT_KIND = {"type": "string", "default": "abstract"}
    ATT_NAME = {"type": "string", "default": lambda: "olympus_{}".format(generate_id())}
    ATT_PATH = {"type": "string", "default": __scratch__}

    def __init__(self, *args, **kwargs):
        Object.__init__(self, *args, **kwargs)

    def __repr__(self):
        return f"<Database (name={self.name}, kind={self.kind})>"

    @property
    def abstract_methods(self):
        return self.ABSTRACT_METHODS

    @property
    def db_exists(self):
        return os.path.isfile(self.file_name)

    @property
    def file_name(self):
        return os.path.join(self.path, "{}.{}".format(self.name, self.file_type))

    @property
    def file_types(self):
        return self.ATT_FILE_TYPE["valid"]

    def _from_file_name(self, file_name):
        self.path = "/".join(file_name.split("/")[:-1])
        self.name = file_name.split("/")[-1].split(".")[0]
        self.file_type = file_name.split("/")[-1].split(".")[1]

    def from_file(self, file_name):
        self._from_file_name(file_name)
        if not self.db_exists:
            Logger.log("Could not find database file {}".format(file_name), "ERROR")
            return None
        self._load_db()

    def get_campaigns(self):
        return self._get_campaigns()

    def list_campaigns(self):
        return self._list_campaign_ids()

    def log_campaign(self, campaign):
        self._log_campaign(campaign)

    ABSTRACT_METHODS = locals()


# ===============================================================================
