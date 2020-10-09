#!/usr/bin/env python

import numpy as np
import pytest
from olympus import Observations
from olympus import Planner, Emulator
from olympus.planners.abstract_planner import AbstractPlanner
from olympus.planners import planner_names
from olympus import ParameterVector


@pytest.mark.parametrize("planner_kind", planner_names)
def test_planners_ask_tell(two_param_space, planner_kind):
    """Tests the function Planner
    """
    planner = Planner(kind=planner_kind)
    planner.set_param_space(param_space=two_param_space)
    obs = Observations()
    planner.tell(observations=obs)
    param = planner.ask()
    value = ParameterVector().from_dict({'objective': 0.})
    obs.add_observation(param, value)
    planner.tell(observations=obs)


@pytest.mark.parametrize("planner_kind", planner_names)
def test_planners_recommend(two_param_space, planner_kind):
    """Tests the recommend method in all planners
    """
    planner = Planner(kind=planner_kind)
    planner.set_param_space(param_space=two_param_space)
    obs = Observations()

    # special case for LatinHypercube: we need to set the number of iterations in advance!
    num_iter = 3
    if planner_kind == 'LatinHypercube':
        planner.budget = num_iter

    for i in range(num_iter):
        param = planner.recommend(observations=obs)
        value = ParameterVector().from_dict({'objective': 0.})
        obs.add_observation(param, value)


@pytest.mark.parametrize("planner_kind", planner_names)
def test_planners_optimize(planner_kind):
    """Tests the method optimize in all Planners
    """
    print('PLANNER_KIND', planner_kind)
    planner = Planner(kind=planner_kind)
    emulator = Emulator(dataset='hplc', model='BayesNeuralNet')
    planner.optimize(emulator=emulator, num_iter=3, verbose=False)


def test_custom_planner_class(two_param_space):
    """Tests using a custom planner that is built by inheriting from AbstractPlanner.
    """
    planner = Planner(kind=MyPlannerAlg)
    planner.set_param_space(param_space=two_param_space)
    obs = Observations()
    planner.tell(observations=obs)
    param = planner.ask()
    value = ParameterVector().from_dict({'objective': 0.})
    obs.add_observation(param, value)
    planner.tell(observations=obs)


class MyPlannerAlg(AbstractPlanner):
    def __init__(self, goal='minimize'):
        """Example of class (implementing a random sampler) that can be passed to Planner to use a custom algorithm.
        """
        AbstractPlanner.__init__(**locals())

    def _set_param_space(self, param_space):
        self._param_space = []
        for param in param_space:
            if param.type == 'continuous':
                param_dict = {'name': param.name, 'type': param.type, 'domain': (param.low, param.high)}
                self._param_space.append(param_dict)

    def _tell(self, observations):
        self._params = observations.get_params(as_array=True)
        self._values = observations.get_values(as_array=True, opposite=self.flip_measurements)

    def _ask(self):
        new_params = []
        for param in self._param_space:
            new_param = np.random.uniform(low=param['domain'][0], high=param['domain'][1])
            new_params.append(new_param)

        param_vector = ParameterVector().from_array(new_params, self.param_space)
        return param_vector



if __name__ == '__main__':
    from olympus.planners import get_planners_list
    print(get_planners_list())


#	test_planners_optimize(get_planners_list())
#    test_planners_optimize('BasinHopping')
#
#    from olympus import Parameter, ParameterSpace
#    param_space = ParameterSpace()
#    param_space.add(Parameter(name='param_0'))
#    param_space.add(Parameter(name='param_1'))
#    test_custom_planner_class(param_space)
