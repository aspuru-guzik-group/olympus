#!/usr/bin/env python

import os
from glob import glob

olympus_home = os.path.dirname(os.path.abspath(__file__))
olympus_scratch = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".scratch"
)
__home__ = os.environ.get("OLYMPUS_HOME") or olympus_home
__scratch__ = os.environ.get("OLYMPUS_SCRATCH") or olympus_scratch
__emulator_path__ = os.environ.get("OLYMPUS_EMULATOR_PATH") or os.path.abspath(
    os.path.join(__home__, "emulators")
)

# ===============================================================================
# Make sure that home and scratch exist
for dir_name in [__home__, __scratch__]:
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

# ===============================================================================

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

# ===============================================================================
# this is where we can read in environment variables to modify
# the behavior of the logger
from .utils.logger import MessageLogger

Logger = MessageLogger()

# ===============================================================================

from .campaigns import Campaign, Observations, ParameterSpace
from .analyzer import Analyzer
from .baseline import Baseline
from .databases import Database
from .datasets import Dataset, list_datasets
from .emulators import Emulator, list_trained_emulators
from .models import Model
from .noises import Noise
from .objects import (
    Object,
    Parameter,
    ParameterCategorical,
    ParameterContinuous,
    ParameterDiscrete,
    ParameterVector,
)
from .olympus import Olympus
from .planners import Planner, list_planners
from .plotter import Plotter
from .surfaces import Surface

# ===============================================================================


# ===============================================================================
