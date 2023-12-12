#!/usr/bin/env python

# ======================================================================

import os

__home__ = os.path.dirname(os.path.abspath(__file__))

# ======================================================================

# from .cv_manager         import CV_Manager
from .emulator import Emulator


# get list of all available emulators
def list_trained_emulators():
    import glob

    emulators = []
    for dir_name in glob.glob(os.path.join(__home__, "emulator_*")):
        if os.path.isfile(dir_name):
            continue
        dir_name = os.path.basename(dir_name)
        dataset_name = dir_name.split("_")[-2]
        model_name = dir_name.split("_")[-1]
        emulators.append(f"{dataset_name}_{model_name}")
    return emulators


# ======================================================================
