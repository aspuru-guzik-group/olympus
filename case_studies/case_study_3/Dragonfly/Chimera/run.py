#!/usr/bin/env python

import os, sys
import pickle
import numpy as np

import olympus
from olympus.datasets import Dataset
from olympus.evaluators import Evaluator
from olympus.emulators import Emulator
from olympus.campaigns import Campaign
from olympus.planners import Planner
from olympus.scalarizers import Scalarizer


sys.path.append('../../../')
from utils import save_pkl_file, load_data_from_pkl_and_continue

#--------
# CONFIG
#--------

dataset_name = 'dye_lasers'
planner_name = 'Dragonfly'
scalarizer_name = 'Chimera'

budget = 200
num_repeats = 40

# check whether we are appending to previous results
data_all_repeats, missing_repeats = load_data_from_pkl_and_continue(num_repeats)

for num_repeat in range(num_repeats):

    print(f'\nTESTING {planner_name} ON {dataset_name} WITH {scalarizer_name} REPEAT {num_repeat} ...\n')

    if dataset_name == 'dye_lasers':
        # fully categorical, lookup table
        dataset = Dataset(kind=dataset_name)

        planner = Planner(kind=planner_name)
        planner.set_param_space(dataset.param_space)

        campaign = Campaign()
        campaign.set_param_space(dataset.param_space)
        campaign.set_value_space(dataset.value_space)

        if scalarizer_name == 'Chimera':
            scalarizer = Scalarizer(
                kind='Chimera', 
                value_space=dataset.value_space,
                goals=['max', 'min', 'max'],
                tolerances=[0.7, 0.2, 0.2],
                absolutes=[True, True, True]
            )
        elif scalarizer_name == 'Parego':
            scalarizer = Scalarizer(
                kind='Parego', 
                value_space=dataset.value_space,
                goals=['max', 'min', 'max'],
                rho=0.05,
            )
        
        elif scalarizer_name == 'WeightedSum':
            scalarizer = Scalarizer(
                kind='WeightedSum', 
                value_space=dataset.value_space,
                goals=['max', 'min', 'max'],
                weights=[3., 2., 1.],
            )
        
        elif scalarizer_name == 'ConstrainedAsf':
            pass
            # TODO: implement this! 
            

        evaluator = Evaluator(
            planner=planner, 
            emulator=dataset,
            campaign=campaign,
            scalarizer=scalarizer,
        )


    evaluator.optimize(num_iter=budget)

    data_all_repeats.append(campaign)
    save_pkl_file(data_all_repeats)
    
    print('Done!')