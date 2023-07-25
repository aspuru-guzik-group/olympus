#!/usr/bin/env python

from olympus import Olympus
from olympus.databases import Database
from olympus import Campaign
from olympus.datasets import Dataset
from olympus.emulators import Emulator
from olympus.evaluators import Evaluator
from olympus.planners import Planner
from olympus.scalarizers import Scalarizer
from olympus.surfaces import Surface

# ===============================================================================


def test_evaluator_surface():
    planner = Planner()
    surface = Surface()
    evaluator = Evaluator(planner=planner, emulator=surface)


def test_evaluator_emulator():
    planner = Planner()
    emulator = Emulator(dataset="agnp", model="BayesNeuralNet")
    evaluator = Evaluator(planner=planner, emulator=emulator)



def test_evaluator_moo():
    planner = Planner()
    emulator = Dataset(kind="dye_lasers")
    scalarizer = Scalarizer(
        kind="Chimera",
        value_space=emulator.value_space,
        goals=["max", "min", "max"],
        tolerances=[0.2, 0.4, 0.6],
        absolutes=[False, False, False],
    )
    evaluator = Evaluator(
        planner=planner, emulator=emulator, scalarizer=scalarizer
    )

    assert evaluator.campaign.is_moo


def test_evaluator_simplex_constraint():
    planner = Planner()
    emulator = Emulator(dataset="photo_wf3", model="BayesNeuralNet")
    campaign = Campaign()
    campaign.set_param_space(emulator.param_space)
    campaign.set_value_space(emulator.value_space)
    print(campaign)

    evaluator = Evaluator(
        planner=planner, emulator=emulator, campaign=campaign
    )
    assert evaluator.emulator.parameter_constriants == "simplex"


def test_evaluator_optimize_surface_cont():
    planner = Planner(kind='Botorch')
    surface = Surface(kind='Branin', param_dim=2)
    evaluator = Evaluator(planner=planner, surface=surface)
    evaluator.optimize(num_iter=10)

def test_evaluator_optimize_surface_cat():
    planner = Planner(kind='Botorch')
    surface = Surface(kind='CatDejong', param_dim=2, num_opts=21)
    evaluator = Evaluator(planner=planner, surface=surface)
    evaluator.optimize(num_iter=10)

def test_evaluator_optimize_emulator_cont():
    #planner = Planner(kind='RandomSearch')
    planner = Planner(kind='Hyperopt')
    emulator = Emulator(dataset='hplc', model='BayesNeuralNet')
    evaluator = Evaluator(planner=planner, emulator=emulator)

    evaluator.optimize(num_iter=10)


def test_evaluator_optimize_emulator_cat():
    #planner = Planner(kind='RandomSearch')
    planner = Planner(kind='Botorch')
    dataset = Dataset(kind='perovskites')
    evaluator = Evaluator(planner=planner, emulator=dataset)

    evaluator.optimize(num_iter=10)

def test_run_cont():
    dataset_name = 'hplc' 
    num_reps = 10  
    num_iter = 20
    planners = ['RandomSearch', 'Hyperopt', 'Botorch'] 

    database = Database()

    for planner in planners:
        for rep in range(num_reps):

            Olympus().run(
                goal='maximize',
                planner=planner,
                dataset=dataset_name,
                campaign=Campaign(),
                database=database,
                num_iter=num_iter
            )


def test_run_cat():
    dataset_name = 'perovskites' 
    num_reps = 10  
    num_iter = 20
    planners = ['RandomSearch', 'Hyperopt', 'Botorch'] 

    database = Database()

    for planner in planners:
        for rep in range(num_reps):

            Olympus().run(
                goal='maximize',
                planner=planner,
                dataset=dataset_name,
                campaign=Campaign(),
                database=database,
                num_iter=num_iter
            )
 



# ===============================================================================

if __name__ == "__main__":
    #test_evaluator_optimize_emulator_cont()
    #test_evaluator_optimize_emulator_cat()
    #test_evaluator_optimize_surface_cont()
    #test_evaluator_optimize_surface_cat()
    
    #test_run_cont()
    test_run_cat()