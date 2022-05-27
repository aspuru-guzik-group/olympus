#!/usr/bin/env python

# ==============================================================================

import traceback

from olympus import Logger

# ==============================================================================

try:
    import sqlalchemy
except ModuleNotFoundError:
    error = traceback.format_exc()
    for line in error.split("\n"):
        if "ModuleNotFoundError" in line:
            module = line.strip().strip("'").split("'")[-1]
    message = """Sqlite databases require {module}, which could not be found. 
	Please install {module} or use a different database backend""".format(
        module=module
    )
    Logger.log(message, "WARNING", only_once=True)
    raise ModuleNotFoundError

# ==============================================================================

from olympus.databases.database_sqlite.sqlite_interface import SqliteInterface
from olympus.databases.database_sqlite.sqlite_operations import (
    AddEntry,
    FetchEntries,
    UpdateEntries,
)
from olympus.databases.database_sqlite.wrapper_sqlite import Wrapper_sqlite
