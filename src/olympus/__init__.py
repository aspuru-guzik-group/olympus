#!/usr/bin/env python

import os
from glob import glob

olympus_home = os.path.dirname(os.path.abspath(__file__))
olympus_scratch = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".scratch")
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
from .utils import MessageLogger

Logger = MessageLogger()

# ===============================================================================

from .objects import Object
from .objects import Parameter
from .objects import ParameterContinuous
from .objects import ParameterVector

from .campaigns import Campaign
from .campaigns import Observations
from .campaigns import ParameterSpace

from .analyzer import Analyzer
from .databases import Database
from .datasets import Dataset
from .emulators import Emulator
from .models import Model
from .planners import Planner
from .plotter import Plotter
from .surfaces import Surface
from .noises import Noise

from .olympus import Olympus

# ===============================================================================

from .datasets import list_datasets
from .emulators import list_trained_emulators
from .planners import list_planners

# ===============================================================================

from .baseline import Baseline
