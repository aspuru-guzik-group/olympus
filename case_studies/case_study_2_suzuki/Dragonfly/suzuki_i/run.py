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

dataset_name = 'suzuki_i'
planner_name = 'Dragonfly'

budget = 200
num_repeats = 40


# check whether we are appending to previous results
data_all_repeats, missing_repeats = load_data_from_pkl_and_continue(num_repeats)


for num_repeat in range(missing_repeats):
        
    print(f'\nTESTING {planner_name} ON {dataset_name} REPEAT {num_repeat} ...\n')
    
    if dataset_name == 'suzuki': 
        
        # fully continuous, emulated dataset
        emulator = Emulator(dataset=dataset_name, model='BayesNeuralNet')
        planner = Planner(kind=planner_name)
        planner.set_param_space(emulator.param_space)

        campaign = Campaign()
        campaign.set_param_space(emulator.param_space)
        campaign.set_value_space(emulator.value_space)

        evaluator = Evaluator(
            planner=planner, 
            emulator=emulator,
            campaign=campaign,
        )
        
    elif dataset_name == 'suzuki_edbo':
        
        # fully categorical, lookup table
        dataset = Dataset(kind=dataset_name)

        planner = Planner(kind=planner_name)
        planner.set_param_space(dataset.param_space)

        campaign = Campaign()
        campaign.set_param_space(dataset.param_space)
        campaign.set_value_space(dataset.value_space)

        evaluator = Evaluator(
            planner=planner, 
            emulator=dataset,
            campaign=campaign,
        )
        
    elif dataset_name in ['suzuki_i', 'suzuki_ii', 'suzuki_iii', 'suzuki_iv']:
        
        # mixed parameter, emulator, multi-objective optimization
        emulator = Emulator(dataset=dataset_name, model='BayesNeuralNet')
        planner = Planner(kind=planner_name)
        planner.set_param_space(emulator.param_space)

        campaign = Campaign()
        campaign.set_param_space(emulator.param_space)
        campaign.set_value_space(emulator.value_space)
        
        scalarizer = Scalarizer(
            kind='Chimera', 
            value_space=emulator.value_space,
            goals=['max', 'max'],
            tolerances=[0.9, 0.0],
            absolutes=[False, False]
        )

        evaluator = Evaluator(
            planner=planner, 
            emulator=emulator,
            campaign=campaign,
            scalarizer=scalarizer,
        )
    
    evaluator.optimize(num_iter=budget)
    
    data_all_repeats.append(campaign)
    save_pkl_file(data_all_repeats)
    
    print('Done!')
