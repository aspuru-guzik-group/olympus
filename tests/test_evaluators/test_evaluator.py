#!/usr/bin/env python

from olympus.evaluators import Evaluator 
from olympus.planners   import Planner
from olympus.surfaces   import Surface 
from olympus.emulators  import Emulator
from olympus.datasets   import Dataset
from olympus.scalarizers import Scalarizer
from olympus import Campaign

#===============================================================================

def test_evaluator_surface():
    planner  = Planner()
    surface  = Surface()
    evaluator = Evaluator(planner=planner, emulator=surface)


def test_evaluator_emulator():
    planner = Planner()
    emulator = Emulator(dataset='agnp', model='BayesNeuralNet')
    evaluator = Evaluator(planner=planner, emulator=emulator)


def test_evaluator_moo():
    planner = Planner()
    emulator = Dataset(kind='dye_lasers')
    scalarizer = Scalarizer(
        kind='Chimera',
        value_space=emulator.value_space,
        goals=['max', 'min', 'max'],
        tolerances=[0.2, 0.4, 0.6],
        absolutes=[False, False, False]
    )
    evaluator = Evaluator(planner=planner, emulator=emulator, scalarizer=scalarizer)

    assert evaluator.campaign.is_moo


def test_evaluator_simplex_constraint():
    planner = Planner()
    emulator = Emulator(dataset='photo_wf3', model='BayesNeuralNet')
    campaign = Campaign()
    campaign.set_param_space(emulator.param_space)
    campaign.set_value_space(emulator.value_space)
    print(campaign)

    evaluator = Evaluator(planner=planner, emulator=emulator, campaign=campaign)
    assert evaluator.emulator.parameter_constriants == 'simplex'



#===============================================================================

if __name__ == '__main__':
    test_evaluator_surface()
