#!/usr/bin/env python

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, JOB_STATE_DONE
from collections import OrderedDict

from olympus.planners import AbstractPlanner
from olympus.objects import ParameterVector


class Hyperopt(AbstractPlanner):

    def __init__(self, goal='minimize', show_progressbar=False):
        """
        Tree of Parzen Estimators (TPE) as implemented in HyperOpt.
        Args:
            goal (str): The optimization goal, either 'minimize' or 'maximize'. Default is 'minimize'.
            show_progressbar (bool): If True, show a progressbar.
        """
        AbstractPlanner.__init__(**locals())
        self._trials   = Trials()  # these is a Hyperopt object that stores the search history
        self._hp_space = None  # these are the params in the Hyperopt format

    def _set_param_space(self, param_space):
        self._param_space = []
        for param in param_space:
            if param.type == 'continuous':
                param_dict = {'name': param.name, 'type': param.type, 'domain': (param.low, param.high)}
            self._param_space.append(param_dict)
        # update hyperopt space accordingly
        self._set_hp_space()

    def _tell(self, observations):
        self._params = observations.get_params(as_array=False)
        self._values = observations.get_values(as_array=True, opposite=self.flip_measurements)
        # update hyperopt Trials accordingly
        self._set_hp_trials()

    def _set_hp_space(self):
        space = []
        # go through all parameters we have defined and convert them to Hyperopt format
        for param in self._param_space:
            if param['type'] == 'continuous':
                space.append((param['name'], hp.uniform(param['name'], param['domain'][0], param['domain'][1])))
        # update instance attribute that is the space input for Hyperopt fmin
        self._hp_space = OrderedDict(space)

    def _set_hp_trials(self):
        self._trials = Trials()
        if self._params is not None and len(self._params) > 0:
            for tid, (param, loss) in enumerate(zip(self._params, self._values)):
                idxs = {k: [tid] for k, v in param.items()}
                vals = {k: [v] for k, v in param.items()}
                hyperopt_trial = Trials().new_trial_docs(
                                        tids=[tid],
                                        specs=[None],
                                        results=[{'loss': loss, 'status': STATUS_OK}],
                                        miscs=[{'tid': tid,
                                                'cmd': ('domain_attachment', 'FMinIter_Domain'),
                                                'idxs': idxs,
                                                'vals': vals,
                                                'workdir': None}]
                                        )
                hyperopt_trial[0]['state'] = JOB_STATE_DONE
                self._trials.insert_trial_docs(hyperopt_trial)
                self._trials.refresh()

    def _ask(self):
        # NOTE: we pass a dummy function as we just ask for the new (+1) set of parameters
        _ = fmin(fn=lambda x: 0, space=self._hp_space, algo=tpe.suggest, max_evals=self.num_generated,
                 trials=self._trials, show_progressbar=self.show_progressbar)

        # make sure the number of parameters asked matches the number of Hyperopt iterations/trials
        assert len(self._trials.trials) == self.num_generated
        # get params from last dict in trials.trials
        proposed_params = self._trials.trials[-1]['misc']['vals']
        for key, value in proposed_params.items():
            proposed_params[key] = value[0]  # this is just to make value not a list

        return ParameterVector(dict=proposed_params, param_space=self.param_space)

    # @staticmethod
    # def propose_randomly(num_proposals, param_space, use_descriptors):
    #     """Randomly generate num_proposals proposals. Returns the numerical
    #     representation of the proposals as well as the string based representation
    #     for the categorical variables

    #     Args:
    #             num_proposals (int): the number of random proposals to generate
    #     """
    #     proposals = []
    #     raw_proposals = []
    #     for propsal_ix in range(num_proposals):
    #         sample = []
    #         raw_sample = []
    #         for param_ix, param in enumerate(param_space):
    #             if param.type == "continuous":
    #                 p = np.random.uniform(param.low, param.high, size=None)
    #                 sample.append(p)
    #                 raw_sample.append(p)
    #             elif param.type == "discrete":
    #                 num_options = int(
    #                     ((param.high - param.low) / param.stride) + 1
    #                 )
    #                 options = np.linspace(param.low, param.high, num_options)
    #                 p = np.random.choice(options, size=None, replace=False)
    #                 sample.append(p)
    #                 raw_sample.append(p)
    #             elif param.type == "categorical":
    #                 options = param.options
    #                 p = np.random.choice(options, size=None, replace=False)
    #                 feat = cat_param_to_feat(param, p, use_descriptors)
    #                 sample.extend(feat)  # extend because feat is vector
    #                 raw_sample.append(p)
    #         proposals.append(sample)
    #         raw_proposals.append(raw_sample)
    #     proposals = np.array(proposals)

    #     return proposals, raw_proposals


# -----------
# DEBUGGING
# -----------
if __name__ == '__main__':
    PARAM_TYPE = "continuous"

    NUM_RUNS = 40

    from olympus.campaigns import Campaign, ParameterSpace
    from olympus.objects import (
        ParameterCategorical,
        ParameterContinuous,
        ParameterDiscrete,
    )
    from olympus.surfaces import Surface


    def surface(x):
        return np.sin(8 * x)


    if PARAM_TYPE == "continuous":
        param_space = ParameterSpace()
        param_0 = ParameterContinuous(name="param_0", low=0.0, high=1.0)
        param_space.add(param_0)

        planner = Hyperopt(goal="minimize")
        planner.set_param_space(param_space)

        campaign = Campaign()
        campaign.set_param_space(param_space)

        BUDGET = 24

        for num_iter in range(BUDGET):

            samples = planner.recommend(campaign.observations)
            print(f"ITER : {num_iter}\tSAMPLES : {samples}")
            # for sample in samples:
            for sample in samples:
                sample_arr = sample.to_array()
                measurement = surface(sample_arr.reshape((1, sample_arr.shape[0])))
                campaign.add_observation(sample_arr, measurement[0])


    elif PARAM_TYPE == "categorical":

        surface_kind = "CatDejong"
        surface = Surface(kind=surface_kind, param_dim=2, num_opts=21)

        campaign = Campaign()
        campaign.set_param_space(surface.param_space)

        planner = Hyperopt(goal="minimize")
        planner.set_param_space(surface.param_space)

        OPT = ["x10", "x10"]

        BUDGET = 442

        for iter in range(BUDGET):

            samples = planner.recommend(campaign.observations)
            print(f"ITER : {iter}\tSAMPLES : {samples}")
            # sample = samples[0]
            sample_arr = samples.to_array()
            measurement = np.array(surface.run(sample_arr))
            campaign.add_observation(sample_arr, measurement[0])

            if [sample_arr[0], sample_arr[1]] == OPT:
                print(f"FOUND OPTIMUM AFTER {iter+1} ITERATIONS!")
                break


    elif PARAM_TYPE == "mixed":

        def surface(params):
            return np.random.uniform()

        param_space = ParameterSpace()
        # continuous parameter 0
        param_0 = ParameterContinuous(name="param_0", low=0.0, high=1.0)
        param_space.add(param_0)

        # continuous parameter 1
        param_1 = ParameterContinuous(name="param_1", low=0.0, high=1.0)
        param_space.add(param_1)

        # categorical parameter 2
        param_2 = ParameterCategorical(name="param_2", options=["a", "b", "c"])
        param_space.add(param_2)

        # categorcial parameter 3
        param_3 = ParameterCategorical(name="param_3", options=["x", "y", "z"])
        param_space.add(param_3)

        campaign = Campaign()
        campaign.set_param_space(param_space)

        planner = Hyperopt(goal="minimize")
        planner.set_param_space(param_space)

        BUDGET = 20

        for iter in range(BUDGET):

            samples = planner.recommend(campaign.observations)
            # sample = samples[0]
            sample_arr = samples.to_array()
            measurement = surface(sample_arr)
            print(
                f"ITER : {iter}\tSAMPLES : {samples}\t MEASUREMENT : {measurement}"
            )
            campaign.add_observation(sample_arr, measurement)
