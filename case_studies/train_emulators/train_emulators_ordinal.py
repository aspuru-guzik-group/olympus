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
	model  = BayesNeuralNet(task='ordinal',**params, out_act=dataset_params[current_dataset]['out_act'])
	emulator = Emulator(
		dataset=current_dataset,
		model=model,
		feature_transform=dataset_params[current_dataset]['feature_transform'],
		target_transform=dataset_params[current_dataset]['target_transform']
	)

	print(f'\n\nPARAMS : {params}\n\n')

	scores = emulator.train()
	loss = -scores['test_acc']

	all_losses.append(loss)
	all_cv_scores.append(scores)
	all_params.append(params)
	all_emulators.append(emulator)
	all_test_indices.append(emulator.dataset.test_indices)

	return {'loss': loss, 'status': STATUS_OK}



# datasets to emulate


dataset_names = ['vapdiff_crystal', ]

dataset_params = {
		'vapdiff_crystal': {'out_act': 'sigmoid', 'feature_transform': 'standardize', 'target_transform': 'identity'},
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
