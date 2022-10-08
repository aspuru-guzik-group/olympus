#!/usr/bin/env python

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import olympus
from olympus import __home__
from olympus.campaigns import Campaign, ParameterSpace
from olympus.datasets import Dataset
from olympus.emulators import Emulator
from olympus.planners import Planner
from olympus.models import BayesNeuralNet

from olympus.objects import (
    ParameterContinuous,
    ParameterDiscrete,
    ParameterOrdinal,
    ParameterCategorical
)

dataset = Dataset(kind='mock_ordinal_emulator')
#dataset = Dataset(kind='suzuki')
print(dataset.param_space)
print(dataset.value_space)
print(dataset.task)


#--------------------
# BUILD NEW EMULATOR
#--------------------

# params = {
#     'task': 'ordinal',
#     'batch_size': 50,
#     'hidden_act': 'leaky_relu',
#     'hidden_depth': 4,
#     'hidden_nodes': 42,
#     'learning_rate': 0.001,
#     'reg': 0.01,
#     'out_act': 'sigmoid',
# }

# model = BayesNeuralNet(**params)


# emulator = Emulator(
#     dataset=dataset,
#     model=model,
#     feature_transform='standardize',
#     target_transform='identity',
# )

# emulator.train()
# emulator.save(f'{__home__}/emulators/emulator_mock_ordinal_emulator_BayesNeuralNet')

# # prediction
# params = dataset.test_set_features.to_numpy()[0]
# print(params)

# pred = emulator.run(
# 	params, 
# 	return_paramvector=True, 
# 	return_ordinal_label=True,
# )

# print(pred)


#-------------------------
# LOAD EMULATOR FROM DISK
#-------------------------


emulator = Emulator(dataset='mock_ordinal_emulator', model='BayesNeuralNet')
#emulator = Emulator(dataset='suzuki', model='BayesNeuralNet')

planner = Planner(kind='RandomSearch', goal='maximize')
planner.set_param_space(dataset.param_space)

campaign = Campaign()
campaign.set_param_space(dataset.param_space)
campaign.set_value_space(dataset.value_space)



params = planner.recommend(campaign.observations)
print('params : ', params)

measurement = emulator.run(params) 
print('measurement : ', measurement)