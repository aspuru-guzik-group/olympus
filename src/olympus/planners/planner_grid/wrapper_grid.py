#!/usr/bin/env python

import numpy as np

from olympus import Logger
from olympus.objects import ParameterVector
from olympus.planners.abstract_planner import AbstractPlanner


class Grid(AbstractPlanner):

    PARAM_TYPES = ["continuous", "discrete", "categorical"]

    def __init__(
        self,
        goal="minimize",
        levels=2,
        budget=None,
        exceed_budget=True,
        shuffle=False,
        random_seed=None,
    ):
        """Grid search.

        Note that the number of samples grow exponentially with the number of dimensions. E.g. for a 2-dimensional
        parameter space, with 2 levels, the grid will contain 4 samples; for a 3-dimensional space, it will contain
        8 samples; for a 6-dimensional space, 64 samples.

        Args:
            goal (str): The optimization goal, either 'minimize' or 'maximize'. Default is 'minimize'.
            levels (int, list): How many locations in parameter space to sample per parameter/dimension. If an integer
                is provided, the same number of levels will be used for each dimension. Provide a list to use different
                levels for different dimensions. Default is 2.
            budget (int): Maximum number of samples you want to evaluate. From the specified ``budget`` an adequate
                value for the levels will be derived. Note that, if provided, the argument ``levels`` will be discarded.
                Default is None.
            exceed_budget (bool): Whether to allow building the grid with more samples then ``budget``.
                This means some points might not be evaluated, but ensures there will be enough grid points to run as
                many evaluations as defined in ``budget``. If False, the number of grid points will be less or equal to
                ``budget``; this guarantees the budget is enough to guarantee the exploration of the whole grid.
                Default is True.
            shuffle (bool): Whether to randomize the order of the samples in the grid. Default is False.
            random_seed (int): Random seed. Set a random seed for a reproducible randomization of the grid if ``shuffle``
                was set to True.
        """
        AbstractPlanner.__init__(**locals())
        self.grid_created = False

    def reset(self):
        """Clears the remaining samples in the grid and prepares the planner for re-initialisation."""
        self.samples_loc = None
        self.samples = None
        self.grid_created = False

    def _set_param_space(self, param_space):
        self.param_space = param_space
        self.dims = len(self.param_space)
        self.bounds = self.param_space.param_bounds

        # if budget provided, define levels
        if self.budget is not None:
            self._get_approximate_levels()

        # allow providing a list of levels to tune the budget
        if isinstance(self.levels, int):
            self._levels = [self.levels] * self.dims
        elif isinstance(self.levels, list) or isinstance(
            self.levels, np.ndarray
        ):
            if len(self.levels) != self.dims:
                message = (
                    f"The number of levels provided ({len(self.levels)}) does not match dimensionality of the "
                    f"parameter space ({self.dimlists})."
                )
                Logger.log(message, "ERROR")
            self._levels = list(self.levels)
        else:
            raise ValueError(
                "Argument `level` can only be a integer or a list."
            )

    def _tell(self, observations):
        # grid search does not care about previous observations
        pass

    def _create_grid(self):
        self.samples_loc = []  # sample locations by dimension

        # define location of grid samples for each dimension
        for param, level in zip(self.param_space, self._levels):
            if param.type == "continuous":
                loc = np.linspace(start=param.low, stop=param.high, num=level)
            elif param.type == "discrete":
                num_options = int(
                    ((param.high - param.low) / param.stride) + 1
                )
                options = np.linspace(param.low, param.high, num_options)
                tiled_options = np.tile(options, (level // num_options) + 1)
                if self.shuffle:
                    np.random.shuffle(tiled_options)
                loc = tiled_options[:level]
            elif param.type == "categorical":
                options = param.options
                num_options = len(param.options)
                tiled_options = np.tile(options, (level // num_options) + 1)
                loc = tiled_options[:level]

            self.samples_loc.append(loc)

        meshgrid = np.stack(
            np.meshgrid(*self.samples_loc), len(self.samples_loc)
        )  # make grid
        num_samples = np.prod(
            np.shape(meshgrid)[:-1]
        )  # number of samples in grid

        # all grid samples in a 2D array
        self.samples = np.reshape(
            meshgrid, newshape=(num_samples, len(self.samples_loc))
        )
        # shuffle is we are sampling these points at random
        if self.shuffle is True:
            np.random.seed(self.random_seed)
            np.random.shuffle(self.samples)
        self.samples = list(self.samples)
        self.grid_created = True

    def _get_approximate_levels(self):
        # initialise with lowest number of samples that does not exceed budget
        base_level = int(np.floor(self.budget ** (1.0 / self.dims)))
        self.levels = np.array([base_level] * self.dims)
        effective_budget = np.prod(
            self.levels
        )  # actual number of samples created
        excess = int(self.budget - effective_budget)  # excess budget left

        # now add one level at a time to each dimension
        counter = 0
        while (
            excess > 0
        ):  # add levels until we exhaust all excess budget samples
            # if we want to have samples <= budget (i.e. do not exceed budget)
            if self.exceed_budget is False:
                # add 1 to last level as this should never get to receive a +1 anyway
                effective_budget_lookahead = np.prod(self.levels[:-1]) * (
                    self.levels[-1] + 1
                )
                if effective_budget_lookahead > self.budget:
                    break
            # increment one level at a time
            i = counter % self.dims
            self.levels[i] += 1
            counter += 1
            # update effective budget and excess
            effective_budget = np.prod(self.levels)
            excess = int(self.budget - effective_budget)

        if excess >= 0:
            excess_type = "budgeted evaluations"
        else:
            excess_type = "grid points"
        message = (
            f"Budget provided, discarding argument `levels`. Given a budget of {self.budget} and a "
            f"parameter space of dimensionality {self.dims}, we will explore {self.levels} levels. "
            f"Given an excess of {np.abs(excess)} {excess_type}, the effective number "
            f"of evaluations required is {effective_budget}."
        )
        Logger.log(message, "INFO")

    def _ask(self):
        if self.grid_created is False:
            self._create_grid()

        param = self.samples.pop(0)

        olymp_param = []
        for param_ix, suggestion in enumerate(param):
            if self.param_space[param_ix] in ["continuous", "discrete"]:
                olymp_param.append(np.float(suggestion))
            else:
                olymp_param.append(suggestion)

        if len(self.samples) == 0:
            message = "Last parameter being provided - there will not be any more available samples in the grid."
            Logger.log(message, "INFO")

        return ParameterVector(array=param, param_space=self.param_space)


# DEBUG:
if __name__ == "__main__":

    from olympus import Campaign
    from olympus.datasets import Dataset

    BUDGET = 100

    d = Dataset(kind="perovskites")

    planner = Grid(goal="minimize", budget=BUDGET, shuffle=True)
    planner.set_param_space(d.param_space)

    campaign = Campaign()
    campaign.set_param_space(d.param_space)

    for i in range(BUDGET):
        print(f"ITERATION : ", i)

        sample = planner.recommend(campaign.observations)
        print("SAMPLE : ", sample)

        measurement = d.run([sample], return_paramvector=False)[0]
        print("MEASUREMENT : ", measurement)

        campaign.add_observation(sample, measurement)
