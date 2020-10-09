#!/usr/bin/env python


import os
__home__ = os.path.dirname(os.path.abspath(__file__))

#===============================================================================

from .abstract_database import AbstractDatabase
from .database          import Database

import glob
db_types  = []
databases = {}
for dir_name in glob.glob('{}/database_*'.format(__home__)):
    dir_name = dir_name.split('/')[-1][9:]
    db = Database(kind=dir_name)
    db_types.extend(db.file_types)
    databases[dir_name.lower()] = db

#===============================================================================
