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


search_space = {
	'batch_size': hp.quniform('batch_size', 10, 100, 10),
	'hidden_act': hp.choice('hidden_act', ['leaky_relu'] ),
	'hidden_depth': hp.quniform('hidden_depth', 2, 5, 1),
	'hidden_nodes': hp.quniform('hidden_nodes', 28, 96, 4),
	'learning_rate': hp.uniform('learning_rate', 1e-5, 5e-3),
	'reg': hp.uniform('reg', 0.0001, 0.5)
}
int_params = ['batch_size', 'hidden_depth', 'hidden_nodes']



def objective(params):
	# build emualtor
	for param, val in params.items():
		if param in int_params:
			params[param] = int(val)
	model  = BayesNeuralNet(task='regression',**params, out_act=dataset_params[current_dataset]['out_act'])
	emulator = Emulator(
		dataset=current_dataset,
		model=model,
		feature_transform=dataset_params[current_dataset]['feature_transform'],
		target_transform=dataset_params[current_dataset]['target_transform']
	)

	scores = emulator.train()
	loss = -scores['test_r2']


	all_losses.append(loss)
	all_cv_scores.append(scores)
	all_params.append(params)
	all_emulators.append(emulator)
	all_test_indices.append(emulator.dataset.test_indices)

	return {'loss': loss, 'status': STATUS_OK}



# datasets to emulate

dataset_names = [
	'oer_plate_4098', 'oer_plate_3851', 'oer_plate_3860', 'oer_plate_3496',
	'p3ht', 'agnp',
	'thin_film', 'crossed_barrel', 'autoam',
	'suzuki_i', 'suzuki_ii', 'suzuki_iii', 'suzuki_iv',
]

# dataset_names = ['oer_plate_4098', 'oer_plate_3851', 'oer_plate_3860', 'oer_plate_3496',
#         'p3ht', 'suzuki_i', 'suzuki_ii', 'suzuki_iii', 'suzuki_iv']

dataset_params = {
		'oer_plate_4098': {'out_act': 'relu', 'feature_transform': 'standardize', 'target_transform': 'normalize'},
		'oer_plate_3851': {'out_act': 'relu', 'feature_transform': 'standardize', 'target_transform': 'normalize'},
		'oer_plate_3860': {'out_act': 'relu', 'feature_transform': 'standardize', 'target_transform': 'normalize'},
		'oer_plate_3496': {'out_act': 'relu', 'feature_transform': 'standardize', 'target_transform': 'normalize'},
		#
		'p3ht': {'out_act': 'relu', 'feature_transform': 'standardize', 'target_transform': 'normalize'},
		'thin_film': {'out_act': 'relu', 'feature_transform': 'identity', 'target_transform': 'normalize'},
		'crossed_barrel': {'out_act': 'relu', 'feature_transform': 'standardize', 'target_transform': 'normalize'},
		'autoam': {'out_act': 'sigmoid', 'feature_transform': 'standardize', 'target_transform': 'normalize'},
		'agnp': {'out_act': 'sigmoid', 'feature_transform': 'standardize', 'target_transform': 'normalize'},
		#
		'suzuki_i': {'out_act': 'relu', 'feature_transform': 'standardize', 'target_transform': 'normalize'},
		'suzuki_ii': {'out_act': 'relu', 'feature_transform': 'standardize', 'target_transform': 'normalize'},
		'suzuki_iii': {'out_act': 'relu', 'feature_transform': 'standardize', 'target_transform': 'normalize'},
		'suzuki_iv': {'out_act': 'relu', 'feature_transform': 'standardize', 'target_transform': 'normalize'},
}


best_scores = {}

for dataset_name in dataset_names:

	current_dataset = dataset_name
	print('CURRENT DATASET : ', current_dataset)

	all_emulators = []
	all_losses = []
	all_cv_scores = []
	all_params = []
	all_test_indices = []


	trials = Trials()

	best = fmin(
		fn=objective,
		space=search_space,
		algo=tpe.suggest,
		max_evals=45,
		trials=trials
	)

	best_idx = np.argmin(all_losses)
	best_emulator = all_emulators[best_idx]

	best_emulator.save(f'emulator_{current_dataset}_BayesNeuralNet')

	best_scores[current_dataset] = {
				'scores':all_cv_scores,
				#'emulators': all_emulators,
				'params': all_params,
				'losses': all_losses,
				'all_test_indices': all_test_indices,
		}
	pickle.dump(best_scores, open('best_scores.pkl', 'wb'))
