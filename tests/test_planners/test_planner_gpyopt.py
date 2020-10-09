#!/usr/bin/env python

import pytest
from olympus import Observations, ParameterVector
from olympus.planners import Gpyopt


# use parametrize to test multiple configurations of the planner
@pytest.mark.parametrize("batch_size, exact_eval, model_type, acquisition_type",
						 [(1, True, 'GP_MCMC', 'EI_MCMC'),
						  (1, False, 'GP_MCMC', 'EI_MCMC'),
						  (2, True, 'GP_MCMC', 'EI_MCMC'),
						  (2, False, 'GP_MCMC', 'EI_MCMC')])
def test_planner_ask_tell(two_param_space, batch_size, exact_eval, model_type, acquisition_type):
	planner = Gpyopt(batch_size=batch_size, exact_eval=exact_eval, model_type=model_type,
					 acquisition_type=acquisition_type)
	planner.set_param_space(param_space=two_param_space)
	param = planner.ask()
	value = ParameterVector().from_dict({'objective': 0.})
	obs = Observations()
	obs.add_observation(param, value)
	planner.tell(observations=obs)

#===============================================================================

if __name__ == '__main__':
	test_planner_ask_tell()
