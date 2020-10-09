#!/usr/bin/env python

from olympus.evaluators import Evaluator 
from olympus.planners   import Planner
from olympus.surfaces   import Surface 

#===============================================================================

def test_evaluator_surface():
    planner  = Planner()
    surface  = Surface()
    evaluator = Evaluator(planner=planner, emulator=surface)


#===============================================================================

if __name__ == '__main__':
    test_evaluator_surface()
