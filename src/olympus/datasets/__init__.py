#!/usr/bin/env python

import os
import traceback

__home__ = os.path.dirname(os.path.abspath(__file__))

# ===============================================================================

from olympus import Logger

# ===============================================================================

# check for pandas
try:
    import pandas
except ModuleNotFoundError:
    error = traceback.format_exc()
    for line in error.split("\n"):
        if "ModuleNotFoundError" in line:
            module = line.strip().strip("'").split("'")[-1]
    message = """Datasets requires {module}, which could not be found.
    Install {module} for accessing datasets directly or providing custom
    datasets""".format(
        module=module
    )
    Logger.log(message, "FATAL")
finally:
    from .dataset import Dataset, load_dataset

# ===============================================================================

import glob

datasets_list = []
for dir_name in glob.glob(f"{__home__}/dataset_*"):
    dir_name = dir_name.split("/")[-1][8:]
    datasets_list.append(dir_name)


def list_datasets():
    return sorted(datasets_list).copy()
