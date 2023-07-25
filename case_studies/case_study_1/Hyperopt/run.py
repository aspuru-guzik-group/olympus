#!/usr/bin/env python

from collections import OrderedDict

import pickle
import numpy as np
from olympus import Olympus
from olympus.campaigns import Campaign, ParameterSpace
from olympus.databases import Database
from olympus.datasets import Dataset

from hyperopt import JOB_STATE_DONE, STATUS_OK, Trials, fmin, hp, tpe

from olympus.objects import ParameterVector, ParameterCategorical


dataset = Dataset(kind='suzuki_edbo')
olymp_param_space = dataset.param_space

def objective(params):
    measurement = dataset.run(
        [
            params['electrophile'],
            params['nucleophile'],
            params['base'],
            params['ligand'],
            params['solvent'],
        ]
    )
    return -measurement[0][0]


num_ind_runs = 40
num_iter = 200

all_runs = []

for run_ix in range(num_ind_runs):

    print(f'Commencing run {run_ix+1} of {num_ind_runs}')
    trials = Trials()

    hyperopt_space = []
    for param in olymp_param_space:
        hyperopt_space.append(
            (param.name, hp.choice(param.name, param.options))
        )

    hyperopt_space = OrderedDict(hyperopt_space)


    best = fmin(
        fn=objective,
        space=hyperopt_space,
        algo=tpe.suggest,
        max_evals=num_iter,
        trials=trials,
        show_progressbar=True,
    )

    print(best)

    print(trials.trials[-1])
    print(len(trials.trials))


    params = []
    values  = []

    for trial in trials.trials:

        values.append(-trial['result']['loss'])
        param_vec = []
        for param in olymp_param_space:
            ix = trial['misc']['vals'][param.name][0]
            str_ = param.options[ix]
            param_vec.append(str_)
        params.append(param_vec)



    params = np.array(params)
    values = np.array(values)

    # print(params.shape, values.shape)
    # print(params)

    all_runs.append({'params': params, 'values': values})


pickle.dump(all_runs, open('results.pkl', 'wb'))
