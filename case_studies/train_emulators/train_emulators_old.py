#!/usr/bin/env python

import pickle
import numpy as np
import pandas as pd


import olympus
from olympus import __home__
from olympus.datasets import Dataset
from olympus.emulators import Emulator
from olympus.models import BayesNeuralNet


from sklearn.metrics import r2_score, mean_squared_error







# datasets to emulate

dataset_names = [
	'alkox', 'benzylation', 'colors_bob', 'colors_n9', 'fullerenes', 'hplc',
	'photo_pce10', 'photo_wf3', 'snar', 'suzuki',
]

model_param_attrs = ['hidden_depth', 'hidden_act', 'out_act', 'learning_rate', 'pred_int', 'reg', 'es_patience', 'max_epochs', 'batch_size']
emulator_param_attrs = ['feature_transform', 'target_transform']




best_scores = {}

for dataset_name in dataset_names:

	current_dataset = dataset_name
	print('CURRENT DATASET : ', current_dataset)

	all_emulators = []
	all_losses = []
	all_cv_scores = []
	all_params = []
	all_test_indices = []


	# load in the original emulator model
	emulator_model = pickle.load(open(f'{__home__}/emulators/emulator_{current_dataset}_BayesNeuralNet/emulator.pickle', 'rb'))

	emulator_params = {e:emulator_model.__dict__[e] for e in emulator_param_attrs}
	model_params = {m:emulator_model.model.__dict__[m] for m in model_param_attrs}

	old_model_scores = emulator_model.__dict__['model_scores']


	# make a new emulator model
	model  = BayesNeuralNet(task='regression', **model_params)
	emulator = Emulator(
		dataset=current_dataset,
		model=model,
		feature_transform=emulator_params['feature_transform'],
		target_transform=emulator_params['target_transform']
	)

	print(emulator.task)
	print(emulator.model.task)

	scores = emulator.train()

	emulator.save(f'emulators_old/emulator_{current_dataset}_BayesNeuralNet')

	best_scores[current_dataset] = {
				'scores':scores,
				'old_scores': old_model_scores
		}
	pickle.dump(best_scores, open('emulators_old/best_scores.pkl', 'wb'))


	print('\n\n')
