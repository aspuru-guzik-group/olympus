#!/usr/bin/env python

import pickle
import numpy as np 
import pandas as pd

from hyperopt import tpe, hp, fmin, STATUS_OK,Trials
from hyperopt.pyll.base import scope

import olympus
from olympus.datasets import Dataset 
from olympus.emulators import Emulator
from olympus.models import BayesNeuralNet


from sklearn.metrics import r2_score, mean_squared_error




def objective(params):
    ''' average performance over cross-validation folds '''
    # build emualtor
    print('CURRENT DATASET : ', current_dataset)
    # for param, val in params.items():
    #     if param in int_params:
    #         params[param] = int(val)
    # model  = BayesNeuralNet(**params, out_act='linear')
    # emulator = Emulator(
    #     dataset='lnp', 
    #     model=model,
    #     feature_transform='standardize',
    #     target_transform='normalize'
    # )

    # cv_scores = emulator.cross_validate()
    # loss = np.mean(cv_scores['validate_rmsd'])
    
    # all_losses.append(loss)
    # all_cv_scores.append(cv_scores)
    # all_params.append(params)
    # all_emulators.append(emulator)
    
    # return {'loss': loss, 'status': STATUS_OK}
    return None



# datasets to emulate

dataset_names = [
	'oer_plate_a', 'oer_plate_b', 'oer_plate_c', 'oer_plate_d',
	'p3ht', 'agnp', 'thin_films', 'crossed_barrel', 'autoam', 
	'suzuki_i', 'suzuki_ii', 'suzuki_iii', 'suzuki_iv',
]

dataset_params = { }

for dataset_name in dataset_names:

	current_dataset = dataset_name
	objective(params=None)

