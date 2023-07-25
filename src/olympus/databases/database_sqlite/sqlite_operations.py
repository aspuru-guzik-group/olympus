#!/usr/bin/env python

# ===============================================================================

import time

import sqlalchemy as sql

from olympus.utils import thread

# ===============================================================================


class AddEntry:
    def __init__(self, db, table, entry):
        self.db = db
        self.table = table
        self.entry = entry

    def execute(self):
        with self.db.connect() as conn:
            conn.execute(self.table.insert(), self.entry)
            conn.close()


# ===============================================================================


class FetchEntries:
    def __init__(self, db, table, select, name="name"):
        self.db = db
        self.table = table
        self.select = select
        self.name = name

        self.entries = None
        self.executed = False
        self.entries_fetched = False

    def execute(self):
        with self.db.connect() as conn:
            selected = conn.execute(self.select)
            entries = selected.fetchall()
            conn.close()
        self.entries = entries
        self.executed = True

    def get_entries(self):
        iter_index = 0
        while not self.executed:
            time.sleep(0.1)
        self.entries_fetched = True
        return self.entries


# ===============================================================================


class UpdateEntries:
    def __init__(self, db, table, updates):
        self.db = db
        self.table = table
        self.updates = updates

    def execute(self):
        with self.db.connect() as conn:
            updated = conn.execute(self.updates)
            conn.close()
