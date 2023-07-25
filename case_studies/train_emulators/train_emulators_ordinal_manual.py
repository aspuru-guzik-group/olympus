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



params = {
		'batch_size': 50,
		'hidden_act': 'leaky_relu',
		'hidden_depth': 5,
		'hidden_nodes': 64,
		'learning_rate': 8e-5,
		'reg': 0.0001,
		'out_act': 'sigmoid',
}

# [INFO] Performance statistics based on transformed data [standardize, identity]:
# [INFO] Train ACC   Score: 0.8400
# [INFO] Test  ACC   Score: 0.8200
# [INFO] Train RMSD   Score: 0.2374
# [INFO] Test  RMSD   Score: 0.2416
#
# [INFO] Performance statistics based on original data:
# [INFO] Train ACC   Score: 0.8692
# [INFO] Test  ACC   Score: 0.8207
# [INFO] Train RMSD   Score: 0.2098
# [INFO] Test  RMSD   Score: 0.2256


model  = BayesNeuralNet(task='ordinal',**params)
emulator = Emulator(
	dataset='vapdiff_crystal',
	model=model,
	feature_transform='standardize',
	target_transform='identity',
)


scores = emulator.train()
print(emulator)
print(scores)


best_emulator = emulator
best_emulator.save(f'emulators_ordinal/emulator_vapdiff_crystal_BayesNeuralNet')
