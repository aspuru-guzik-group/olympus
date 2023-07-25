#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import olympus
from olympus.datasets import Dataset
from olympus.evaluators import Evaluator
from olympus.emulators import Emulator
from olympus.campaigns import Campaign
from olympus.planners import Planner
from olympus.scalarizers import Scalarizer


cs1_datasets = ['redoxmers']
cs1_planners = [
    #'RandomSearch', 
    #'Genetic',
    #'Hyperopt', 
    #'Gpyopt', 
    'Gryffin', 
    #'Dragonfly', 
    #'Botorch',
    #'Smac',
    #'Hebo',
]  


for dataset_name in cs1_datasets:
    for planner_name in cs1_planners:
        
        print(f'\nTESTING {planner_name} ON {dataset_name} ...\n')
            
        if dataset_name == 'dye_lasers':
            # fully categorical, lookup table
            dataset = Dataset(kind=dataset_name)

            planner = Planner(kind=planner_name)
            planner.set_param_space(dataset.param_space)

            campaign = Campaign()
            campaign.set_param_space(dataset.param_space)
            campaign.set_value_space(dataset.value_space)
            
            scalarizer = Scalarizer(
                kind='Chimera', 
                value_space=dataset.value_space,
                goals=['max', 'min', 'max'],
                tolerances=[0.5, 0.5, 0.5],
                absolutes=[False, False, False]
            )

            evaluator = Evaluator(
                planner=planner, 
                emulator=dataset,
                campaign=campaign,
                scalarizer=scalarizer,
            )
        
        elif dataset_name == 'redoxmers':
            # fully categorical, lookup table
            dataset = Dataset(kind=dataset_name)

            planner = Planner(kind=planner_name)
            planner.set_param_space(dataset.param_space)

            campaign = Campaign()
            campaign.set_param_space(dataset.param_space)
            campaign.set_value_space(dataset.value_space)
            
            scalarizer = Scalarizer(
                kind='Chimera', 
                value_space=dataset.value_space,
                goals=['min', 'min', 'min'],
                tolerances=[0.5, 0.5, 0.5],
                absolutes=[False, False, False]
            )

            evaluator = Evaluator(
                planner=planner, 
                emulator=dataset,
                campaign=campaign,
                scalarizer=scalarizer,
            )
            
        evaluator.optimize(num_iter=15)
        
        print('Done!')
