#!/usr/bin/env python

import pickle
from olympus import Olympus
from olympus.campaigns import Campaign
from olympus.databases import Database

olymp = Olympus()
planners = [
    'RandomSearch',
    'Botorch', 'Gryffin',
    # 'Hyperopt', 'Smac', 'Genetic'
]
database = Database(kind='sqlite')

olymp.benchmark(
        dataset='suzuki_edbo',
        planners=planners,
        database=database,
        num_ind_runs=40,
        num_iter=200,
)


pickle.dump(database, open('results.pkl', 'wb'))
