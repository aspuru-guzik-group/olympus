#!/usr/bin/env python

import os
import pickle

from olympus import Database, Logger

# ===============================================================================


# ===============================================================================


def test_auto_init_pkl():
    file_name = "test.pkl"
    with open(file_name, "wb") as content:
        pickle.dump({}, content)
    database = Database().from_file(file_name)
    assert database.db.kind == "pkl"
    os.remove(file_name)
    Logger.purge()


def test_auto_init_pickle():
    file_name = "test.pickle"
    with open(file_name, "wb") as content:
        pickle.dump({}, content)
    database = Database().from_file(file_name)
    assert database.db.kind == "pkl"
    os.remove(file_name)
    Logger.purge()


def test_init_pickle():
    database = Database(kind="pickle")
    assert database.db.kind == "pkl"
    Logger.purge()


def test_init_wrong_path():
    database = Database().from_file("test.pkl")
    assert len(Logger.ERRORS) == 1
    Logger.purge()


def test_init_wrong_type():
    with open("test.dat", "w") as content:
        content.write("olympus")
    database = Database().from_file("test.dat")
    assert len(Logger.ERRORS) == 2
    Logger.purge()
    os.remove("test.dat")
