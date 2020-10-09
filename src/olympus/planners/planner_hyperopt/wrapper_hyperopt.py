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
