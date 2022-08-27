#!/usr/bin/env python

import pickle
from olympus import Olympus
from olympus.campaigns import Campaign
from olympus.databases import Database

olymp = Olympus()
planners = [
    'Gryffin'
]
database = Database(kind='sqlite')

olymp.benchmark(
        dataset='suzuki_edbo',
        planners=planners,
        database=database,
        num_ind_runs=40,
        num_iter=200,
)


observations = [campaign.observations for campaign in database]
pickle.dump(observations, open('results.pkl', 'wb'))
