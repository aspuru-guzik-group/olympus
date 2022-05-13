#!/usr/bin/env python

import numpy as np

from dragonfly.exd.cp_domain_utils import load_config
from dragonfly.exd.experiment_caller import CPFunctionCaller, EuclideanFunctionCaller
from dragonfly import maximise_function, minimise_function
from dragonfly.opt import gp_bandit

from olympus.planners import AbstractPlanner
from olympus.objects import ParameterVector


from dragonfly_utils import infer_problem_type


class Dragonfly(AbstractPlanner):

    PARAM_TYPES = ['continuous', 'discrete', 'categorical']

    def __init__(
        self,
        goal='minimize',
        opt_method='bo',
        random_seed=None,


    ):
        """
        Scalable Bayesian optimization as implemented in the Dragonfly package:
        https://github.com/dragonfly/dragonfly

        Args:
            goal (str): The optimization goal, either 'minimize' or 'maximize'. Default is 'minimize'.
        """
        AbstractPlanner.__init__(**locals())
        # check for and set the random seed
        if not self.random_seed:
            self.random_seed = np.random.randint(1, int(1e7))

        self._has_dragonfly_domain = False
        self._is_dragonfly_built = False

    def _set_param_space(self, param_space):
        self._param_space = []
        for param in param_space:
            if param.type == 'continuous':
                param_dict = {
                    "name": param.name,
                    "type": "float",
                    "min": param.low,
                    "max": param.high,
                }
            elif param.type == 'discrete':
                # discrete numeric
                param_dict = {
                    "name": param.name,
                    "type": "discrete_numeric",
                    "items": f"{param.low}:{param.stride}:{param.high}",
                }
            elif param.type == 'categorical':
                # discrete
                param_dict = {
                    "name": param.name,
                    "type": "discrete",
#                    "items": "-".join([opt for opt in param.options]),
                    "items": param.options,
                }
            else:
                raise NotImplementedError(f'Parameter type {param.type} for {param.name} not implemnted')

            self._param_space.append(param_dict)

        self.problem_type = infer_problem_type(param_space)



    def _build_dragonfly(self):

        config_params = {'name': {"domain": self._param_space}
        config = load_config(config_params)

        print('problem type :', self.problem_type)
        print(self._param_space)
        #domain, domain_orderings = config.domain, config.domain_orderings

        # pick function caller etc. based on the problem type
        if self.problem_type == 'fully_continuous':
            print('HERE!')
            self.func_caller = EuclideanFunctionCaller(None, config.domain)
            self.opt = gp_bandit.EuclideanGPBandit(self.func_caller, ask_tell_mode=True)

        elif self.problem_type in ['fully_categorical', 'fully_discrete']:
            self.func_caller = CPFunctionCaller(None, config.domain, domain_orderings=config.domain_orderings)
            self.opt = gp_bandit.CPGPBandit(self.func_caller, ask_tell_mode=True)

        elif self.problem_type == 'mixed':
            raise NotImplementedError

        else:
            raise NotImplementedError

        self.opt.initialise()

        self._is_dragonfly_built = True





    def _tell(self, observations):
        self._params = observations.get_params()

        self._values = observations.get_values(
            as_array=True, opposite=self.flip_measurements,
        )
        assert len(np.array(self._values).shape)==1

        if not self._is_dragonfly_built:
            self._build_dragonfly()

        if len(self._values) >= 1:
            print(self._params[-1], self._values[-1])

            self.opt.tell([(self._params[-1], self._values[-1])])


    def _ask(self):

        samples = self.opt.ask()

        print(samples)


        return None




#-----------
# DEBUGGING
#-----------


PARAM_TYPE = 'continuous'

NUM_RUNS = 40

from olympus.objects import (
	ParameterContinuous,
	ParameterDiscrete,
	ParameterCategorical,
)
from olympus.campaigns import Campaign, ParameterSpace
from olympus.surfaces import Surface



def surface(x):
	return np.sin(8*x)

if PARAM_TYPE == 'continuous':
	param_space = ParameterSpace()
	param_0 = ParameterContinuous(name='param_0', low=0.0, high=1.0)
	param_space.add(param_0)

	planner = Dragonfly(goal='minimize')
	planner.set_param_space(param_space)

	campaign = Campaign()
	campaign.set_param_space(param_space)

	BUDGET = 24

	for num_iter in range(BUDGET):

		samples = planner.recommend(campaign.observations)
		print(f'ITER : {num_iter}\tSAMPLES : {samples}')
		#for sample in samples:
		sample_arr = samples.to_array()
		measurement = surface(
			sample_arr.reshape((1, sample_arr.shape[0]))
		)
		campaign.add_observation(sample_arr, measurement[0])


elif PARAM_TYPE == 'categorical':

	surface_kind = 'CatDejong'
	surface = Surface(kind=surface_kind, param_dim=2, num_opts=21)

	campaign = Campaign()
	campaign.set_param_space(surface.param_space)

	planner = Botorch(goal='minimize')
	planner.set_param_space(surface.param_space)

	OPT = ['x10', 'x10']

	BUDGET = 442

	for iter in range(BUDGET):

		samples = planner.recommend(campaign.observations)
		print(f'ITER : {iter}\tSAMPLES : {samples}')
		#sample = samples[0]
		sample_arr = samples.to_array()
		measurement = np.array(surface.run(sample_arr))
		campaign.add_observation(sample_arr, measurement[0])

		if [sample_arr[0], sample_arr[1]] == OPT:
			print(f'FOUND OPTIMUM AFTER {iter+1} ITERATIONS!')
			break


elif PARAM_TYPE == 'mixed':

	def surface(params):
		return np.random.uniform()


	param_space = ParameterSpace()
	# continuous parameter 0
	param_0 = ParameterContinuous(name='param_0', low=0.0, high=1.0)
	param_space.add(param_0)

	# continuous parameter 1
	param_1 = ParameterContinuous(name='param_1', low=0.0, high=1.0)
	param_space.add(param_1)

	# categorical parameter 2
	param_2 = ParameterCategorical(name='param_2', options=['a', 'b', 'c'])
	param_space.add(param_2)

	# categorcial parameter 3
	param_3 = ParameterCategorical(name='param_3', options=['x', 'y', 'z'])
	param_space.add(param_3)

	campaign = Campaign()
	campaign.set_param_space(param_space)

	planner = Botorch(goal='minimize')
	planner.set_param_space(param_space)


	BUDGET = 20

	for iter in range(BUDGET):

		samples  = planner.recommend(campaign.observations)
		#sample = samples[0]
		sample_arr = samples.to_array()
		measurement = surface(sample_arr)
		print(f'ITER : {iter}\tSAMPLES : {samples}\t MEASUREMENT : {measurement}')
		campaign.add_observation(sample_arr, measurement)
