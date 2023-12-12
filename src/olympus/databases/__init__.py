#!/usr/bin/env python


import os

__home__ = os.path.dirname(os.path.abspath(__file__))

# ===============================================================================

import glob

from .abstract_database import AbstractDatabase
from .database import Database

db_types = []
databases = {}
for dir_name in glob.glob(os.path.join(__home__, "database_*")):
    dir_name = os.path.basename(dir_name)[9:]
    db = Database(kind=dir_name)
    db_types.extend(db.file_types)
    databases[dir_name.lower()] = db

# ===============================================================================
