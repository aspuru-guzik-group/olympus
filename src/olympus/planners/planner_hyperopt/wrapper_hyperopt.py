#!/usr/bin/env python
from collections import OrderedDict

import numpy as np
from hyperopt import JOB_STATE_DONE, STATUS_OK, Trials, fmin, hp, tpe

from olympus.objects import ParameterVector
from olympus.planners import AbstractPlanner


class Hyperopt(AbstractPlanner):

    PARAM_TYPES = ["continuous", "discrete", "categorical"]

    def __init__(self, goal="minimize", show_progressbar=False):
        """
        Tree of Parzen Estimators (TPE) as implemented in HyperOpt.
        Args:
            goal (str): The optimization goal, either 'minimize' or 'maximize'. Default is 'minimize'.
            show_progressbar (bool): If True, show a progressbar.
        """
        AbstractPlanner.__init__(**locals())
        self._trials = (
            Trials()
        )  # these is a Hyperopt object that stores the search history
        self._hp_space = None  # these are the params in the Hyperopt format

    def _set_param_space(self, param_space):
        self._param_space = []
        for param in param_space:
            if param.type == "continuous":
                param_dict = {
                    "name": param.name,
                    "type": param.type,
                    "domain": (param.low, param.high),
                }
            elif param.type == "discrete":
                param_dict = {
                    "name": param.name,
                    "type": param.type,
                    "domain": (param.low, param.high, param.stirde),
                }
            elif param.type == "categorical":
                param_dict = {
                    "name": param.name,
                    "type": param.type,
                    "options": param.options,
                }
            self._param_space.append(param_dict)
        # update hyperopt space accordingly
        self._set_hp_space()

    def _tell(self, observations):
        self._params = observations.get_params(as_array=False)
        self._values = observations.get_values(
            as_array=True,
            opposite=self.flip_measurements,
        )

        # update hyperopt Trials accordingly
        self._set_hp_trials()

    def _set_hp_space(self):
        space = []
        # go through all parameters we have defined and convert them to Hyperopt format
        for param in self._param_space:
            if param["type"] == "continuous":
                space.append(
                    (
                        param["name"],
                        hp.uniform(
                            param["name"],
                            param["domain"][0],
                            param["domain"][1],
                        ),
                    )
                )
            elif param["type"] == "discrete":
                space.append(
                    (
                        param["name"],
                        hp.quniform(
                            param["name"],
                            param["domian"][0],
                            param["domian"][1],
                            param["domian"][2],
                        ),
                    )
                )
            elif param["type"] == "categorical":
                space.append(
                    (param["name"], hp.choice(param["name"], param["options"]))
                )  # map to integers

        # update instance attribute that is the space input for Hyperopt fmin
        self._hp_space = OrderedDict(space)

    def _set_hp_trials(self):
        self._trials = Trials()
        if self._params is not None and len(self._params) > 0:
            for tid, (param, loss) in enumerate(
                zip(self._params, self._values)
            ):
                idxs = {k: [tid] for k, v in param.items()}
                vals = {k: [v] for k, v in param.items()}
                hyperopt_trial = Trials().new_trial_docs(
                    tids=[tid],
                    specs=[None],
                    results=[{"loss": loss, "status": STATUS_OK}],
                    miscs=[
                        {
                            "tid": tid,
                            "cmd": ("domain_attachment", "FMinIter_Domain"),
                            "idxs": idxs,
                            "vals": vals,
                            "workdir": None,
                        }
                    ],
                )
                hyperopt_trial[0]["state"] = JOB_STATE_DONE
                self._trials.insert_trial_docs(hyperopt_trial)
                self._trials.refresh()

    def _ask(self):

        # print("TRIALS : ", self._trials)

        # if self.num_generated > 1:
        #     print("TRIALS MISC: ", self._trials.trials[-1]["misc"])

        # NOTE: we pass a dummy function as we just ask for the new (+1) set of parameters
        _ = fmin(
            fn=lambda x: 0,
            space=self._hp_space,
            algo=tpe.suggest,
            max_evals=self.num_generated,
            trials=self._trials,
            show_progressbar=self.show_progressbar,
        )
        # make sure the number of parameters asked matches the number of Hyperopt iterations/trials
        assert len(self._trials.trials) == self.num_generated
        # get params from last dict in trials.trials
        proposed_params = self._trials.trials[-1]["misc"]["vals"]

        return_params = {}
        # iterate through the olympus param space
        for param_ix, param in enumerate(self._param_space):
            value = proposed_params[param["name"]]
            if param["type"] == "continuous":
                return_params[param["name"]] = value[0]
            elif param["type"] == "categorical":
                return_params[param["name"]] = param["options"][
                    value[0]
                ]  # reference option

        return ParameterVector(
            dict=return_params, param_space=self.param_space
        )


# DEBUG:
if __name__ == "__main__":

    from olympus import Campaign
    from olympus.datasets import Dataset

    d = Dataset(kind="perovskites")

    planner = Hyperopt(goal="minimize")
    planner.set_param_space(d.param_space)

    campaign = Campaign()
    campaign.set_param_space(d.param_space)

    BUDGET = 30
    for i in range(BUDGET):
        print(f"ITERATION : ", i)

        sample = planner.recommend(campaign.observations)
        print("SAMPLE : ", sample)

        measurement = d.run([sample], return_paramvector=False)[0]
        print("MEASUREMENT : ", measurement)

        campaign.add_observation(sample, measurement)

        print("\n\n\n")
