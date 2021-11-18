#!/usr/bin/env python

import numpy as np
import pandas as pd

import olympus
from olympus import Olympus
from olympus import Database
from olympus import Campaign

from olympus import Planner

from olympus import list_datasets, list_planners

print(list_datasets())
print(list_planners())

NUM_REPS = 20
NUM_ITER = 100

DATASET = 'oled'#'perovskites'
PLANNERS = ['RandomSearch', 'Gpyopt', 'Gryffin']

olymp = Olympus()
database=Database()

for PLANNER in PLANNERS:
    for rep in range(NUM_REPS):
        print(f"Algorithm: {PLANNER} [repetition {rep+1}]")

        olymp.run(
            planner=PLANNER,
            dataset=DATASET,
            campaign=Campaign(),
            database=database,
            num_iter=NUM_ITER,
        )


# collect the campaigns
campaigns = [campaign for campaign in database]

import pickle

pickle.dump(campaigns, open('result_campaigns.pkl', 'wb'))
