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
from olympus.evaluators import Evaluator

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



#----------------------
# TEST RANDOM SAMPLING
#----------------------

# emulator = Emulator(dataset='mock_ordinal_emulator', model='BayesNeuralNet')
# #emulator = Emulator(dataset='suzuki', model='BayesNeuralNet')
# print(emulator)
#
#
# planner = Planner(kind='RandomSearch', goal='maximize')
# planner.set_param_space(dataset.param_space)
#
# campaign = Campaign()
# campaign.set_param_space(dataset.param_space)
# campaign.set_value_space(dataset.value_space)
#

# BUDGET = 10
#
# print('\n\n')
#
# while len(campaign.observations.get_values()) < BUDGET:
#
#     # switch the string-based ordinal values to integer-based
#     campaign.observations_to_int()
#
#     params = planner.recommend(campaign.observations)
#     print('params : ', params)
#
#     # switch back to string-based rep
#     campaign.observations_to_str()
#
#     # # returning the integer label only
#     # measurement = emulator.run(params)[0][0]
#     # print('integer measurement : ', measurement)
#
#     # returning the string label as parameter vector
#     measurement = emulator.run(
#         params,
#         return_paramvector=True,
#     )
#     print('str measurement : ', measurement)
#
#
#     campaign.add_observation(params[0], measurement)
#
#
# print('\n\nparams')
# print(campaign.observations.get_params())
#
# print('\n\nvalues')
# print(campaign.observations.get_values())
#
# # switch to integer representation
# campaign.observations_to_int()
#
# print('\n\nvalues')
# print(campaign.observations.get_values())
# print(campaign.observations.get_values().dtype)
#
# # switch back to string representation
# campaign.observations_to_str()
#
# print('\n\nvalues')
# print(campaign.observations.get_values())
#



# ------------------------
# TEST BAYESIAN OPTIMIZER
# ------------------------


# emulator = Emulator(dataset='mock_ordinal_emulator', model='BayesNeuralNet')
# #emulator = Emulator(dataset='suzuki', model='BayesNeuralNet')
# print(emulator)
#
#
# planner = Planner(kind='Botorch', goal='minimize')
# planner.set_param_space(dataset.param_space)
#
# campaign = Campaign()
# campaign.set_param_space(dataset.param_space)
# campaign.set_value_space(dataset.value_space)
#
#
#
# BUDGET = 25
#
# print('\n\n')
#
# while len(campaign.observations.get_values()) < BUDGET:
#
#     # switch the string-based ordinal values to integer-based
#     campaign.observations_to_int()
#
#     params = planner.recommend(campaign.observations)
#     print('params : ', params)
#
#     # switch back to string-based rep
#     campaign.observations_to_str()
#
#     # # returning the integer label only
#     # measurement = emulator.run(params)[0][0]
#     # print('integer measurement : ', measurement)
#
#     # returning the string label as parameter vector
#     measurement = emulator.run(
#         params,
#         return_paramvector=True,
#     )
#     print('str measurement : ', measurement)
#
#
#     campaign.add_observation(params, measurement)
#
#
# print('\n\nparams')
# print(campaign.observations.get_params())
#
# print('\n\nvalues')
# print(campaign.observations.get_values())
#
# # switch to integer representation
# campaign.observations_to_int()
#
# print('\n\nvalues')
# print(campaign.observations.get_values())
# print(campaign.observations.get_values().dtype)
#
# # switch back to string representation
# campaign.observations_to_str()
#
# print('\n\nvalues')
# print(campaign.observations.get_values())



#-------------------------------------
# TEST HIGH-LEVEL EVALUATOR INTERFACE
#-------------------------------------


emulator = Emulator(dataset='mock_ordinal_emulator', model='BayesNeuralNet')
print(emulator)

planner = Planner(kind='Botorch', goal='minimize')
planner.set_param_space(dataset.param_space)

campaign = Campaign()
campaign.set_param_space(dataset.param_space)
campaign.set_value_space(dataset.value_space)

evaluator = Evaluator(planner=planner, emulator=emulator, campaign=campaign)

BUDGET = 10

evaluator.optimize(num_iter=BUDGET)

print('\n\nparams')
print(campaign.observations.get_params())

print('\n\nvalues')
print(campaign.observations.get_values())
