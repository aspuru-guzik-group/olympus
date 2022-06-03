#!/usr/bin/env python

import gpytorch
import numpy as np
import torch
from botorch.acquisition import ExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.models import MixedSingleTaskGP, SingleTaskGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.optim import (
    optimize_acqf,
    optimize_acqf_discrete,
    optimize_acqf_mixed,
)
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP

from olympus.objects import ParameterVector
from olympus.planners import AbstractPlanner
from olympus import Logger

from .botorch_utils import (  # TODO: sync up these normalize/standardize functions with the; data transformation within Olympus
    cat_param_to_feat,
    create_available_options,
    forward_normalize,
    forward_standardize,
    get_bounds,
    infer_problem_type,
    project_to_olymp,
    propose_randomly,
    reverse_normalize,
    reverse_standardize,
)


class CategoricalSingleTaskGP(ExactGP, GPyTorchModel):

    _num_outputs = 1

    def __init__(self, train_X, train_Y):
        """Single task GP with a categorical kernel based on the Hamming distance

        Note: this kernel is NOT differentiable with respect to the inputs
        """
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=CategoricalKernel(
                ard_num_dims=train_Y.shape[
                    -1
                ]  # ARD for all categorical dimensions
            )
        )
        self.to(train_X)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class Botorch(AbstractPlanner):

    """Wrapper for GP-based Bayesiam optimization with BoTorch

    Args:
            goal (str): the optimization goal, "maximize" or "minimize"
            batch_size (int): number of samples to measure per batch (will be fixed at 1 for now)
            random_seed (int): the random seed to use
            num_initial_design (int): number of points to sample using the initial
                    design strategy
            init_design_strategy (str): the inital design strategy, "random" or "sobol"
    """

    PARAM_TYPES = ["continuous", "discrete", "categorical"]

    def __init__(
        self,
        goal="minimize",
        use_descriptors=False,
        batch_size=1,
        random_seed=None,
        num_init_design=5,
        init_design_strategy="random",
        **kwargs,
    ):
        AbstractPlanner.__init__(**locals())
        self.goal = goal
        self.batch_size = batch_size
        if random_seed is None:
            self.random_seed = np.random.randint(0, int(10e6))
        else:
            self.random_seed = random_seed
        np.random.seed(self.random_seed)
        self.num_init_design = num_init_design
        self.init_design_strategy = init_design_strategy

    def _set_param_space(self, param_space):
        """set the Olympus parameter space (not actually really needed)"""
        # infer the problem type
        self.problem_type = infer_problem_type(self.param_space)

        # make attribute that indicates wether or not we are using descriptors for
        # categorical variables
        if self.problem_type == "fully_categorical":
            descriptors = []
            for p in self.param_space:
                descriptors.extend(p.descriptors)
            if all(d is None for d in descriptors):
                self.has_descriptors = False
            else:
                self.has_descriptors = True
        else:
            self.has_descriptors = False

        # final check: if the user has overwritten the ability to use descriptors, then
        # we set self.has_descriptors to False
        if not self.use_descriptors and self.has_descriptors:
            message = 'Current dataset has descriptors, but user has overidden their use'
            Logger.log(message, 'WARNING')
            self.has_descriptors = False


    def build_train_data(self):
        """build the training dataset at each iteration"""
        target_params = []
        target_values = []
        # adapt the target task params first --> change the categorical params to ohe
        for sample_ix, (targ_param, targ_value) in enumerate(
            zip(self._params, self._values)
        ):
            sample_x = []
            for param_ix, (space_true, element) in enumerate(
                zip(self.param_space, targ_param)
            ):
                if self.param_space[param_ix].type == "categorical":
                    feat = cat_param_to_feat(space_true, element, self.use_descriptors)
                    sample_x.extend(feat)
                else:
                    sample_x.append(float(element))
            target_params.append(sample_x)
            target_values.append(targ_value)
        train_x = np.array(target_params)  # (# target obs, # param dim)
        train_y = np.array(target_values)  # (# target obs, 1)

        self._mins_x = np.array(
            [np.amin(train_x[:, ix]) for ix in range(train_x.shape[1])]
        )
        self._maxs_x = np.array(
            [np.amax(train_x[:, ix]) for ix in range(train_x.shape[1])]
        )

        self._means_y = np.array(
            [np.mean(train_y[:, ix]) for ix in range(train_y.shape[1])]
        )
        # guard against the case where we have std = 0.0
        self._stds_y = np.array(
            [np.std(train_y[:, ix]) for ix in range(train_y.shape[1])]
        )
        self._stds_y = np.where(self._stds_y == 0.0, 1.0, self._stds_y)

        if (
            not self.problem_type == "fully_categorical"
            and not self.has_descriptors
        ):
            # forward transform on features (min max)
            train_x = forward_normalize(train_x, self._mins_x, self._maxs_x)
        # forward transform on targets (standardization)
        train_y = forward_standardize(train_y, self._means_y, self._stds_y)

        # convert to torch tensors and return
        return torch.tensor(train_x).double(), torch.tensor(train_y).double()

    def _tell(self, observations):
        """unpack the current observations from Olympus

        Args:
                observations (obj): observations from Olympus
        """
        self._params = (
            observations.get_params()
        )  # string encodings of categorical params
        self._values = observations.get_values(
            as_array=True, opposite=self.flip_measurements
        )
        # make values 2d if they are not already
        if len(np.array(self._values).shape) == 1:
            self._values = np.array(self._values).reshape(-1, 1)

    def get_cat_dims(self):
        """return list of indices correspondings to the columns of the
        design matrix which should be considered as categorcial features
        """
        dim = 0
        cat_dims = []
        for param in self.param_space:
            if param.type in ["continuous", "discrete"]:
                dim += 1
            elif param.type == "categorical":
                num_opts = len(param.options)
                cat_dims.extend(np.arange(dim, dim + num_opts))
                dim += num_opts
        return cat_dims

    def _ask(self):
        """query the planner for a batch of new parameter points to measure"""

        if len(self._values) < self.num_init_design:
            # sample using initial design strategy
            sample, raw_sample = propose_randomly(1, self.param_space, self.use_descriptors)
            return_params = ParameterVector().from_array(
                raw_sample[0], self.param_space
            )

        else:
            # use GP surrogate to propose the samples
            # get the scaled parameters and values
            self.train_x_scaled, self.train_y_scaled = self.build_train_data()

            # infer the model based on the parameter types
            if self.problem_type == "fully_continuous":
                model = SingleTaskGP(self.train_x_scaled, self.train_y_scaled)
            elif self.problem_type == "mixed":
                # TODO: implement a method to retrieve the categorical dimensions
                cat_dims = self.get_cat_dims()
                model = MixedSingleTaskGP(
                    self.train_x_scaled, self.train_y_scaled, cat_dims=cat_dims
                )
            elif self.problem_type == "fully_categorical":
                # if we have no descriptors, use a Categorical kernel
                # based on the HammingDistance
                if self.has_descriptors:
                    # we have some descriptors, use the Matern kernel
                    model = SingleTaskGP(
                        self.train_x_scaled, self.train_y_scaled
                    )
                else:
                    # if we have no descriptors, use a Categorical kernel
                    # based on the HammingDistance
                    model = CategoricalSingleTaskGP(
                        self.train_x_scaled, self.train_y_scaled
                    )

            # fit the GP
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)

            # get the incumbent point
            f_best_argmin = torch.argmin(self.train_y_scaled)
            f_best_scaled = self.train_y_scaled[f_best_argmin][0].float()
            f_best_raw = self._values[f_best_argmin][0]

            acqf = ExpectedImprovement(
                model, f_best_scaled, objective=None, maximize=False
            )  # always minimization in Olympus

            bounds = get_bounds(self.param_space, self.has_descriptors)
            choices_feat, choices_cat = None, None

            if self.problem_type == "fully_continuous":
                results, _ = optimize_acqf(
                    acq_function=acqf,
                    bounds=bounds,
                    num_restarts=200,
                    q=self.batch_size,
                    raw_samples=1000,
                )
            elif self.problem_type == "mixed":
                results, _ = optimize_acqf_mixed(
                    acq_function=acqf,
                    bounds=bounds,
                    num_restarts=200,
                    q=self.batch_size,
                    raw_samples=1000,
                    fixed_features_list=[],
                )
            elif self.problem_type == "fully_categorical":
                # need to implement the choices input, which is a
                # (num_choices * d) torch.Tensor of the possible choices
                # need to generate fully cartesian product space of possible
                # choices
                choices_feat, choices_cat = create_available_options(
                    self.param_space, self._params, self.use_descriptors
                )

                if self.has_descriptors:
                    choices_feat = forward_normalize(
                        choices_feat.detach().numpy(),
                        self._mins_x,
                        self._maxs_x,
                    )
                    choices_feat = torch.tensor(choices_feat)

                results, _ = optimize_acqf_discrete(
                    acq_function=acqf,
                    q=self.batch_size,
                    max_batch_size=1000,
                    choices=choices_feat,
                    unique=True,
                )

            # convert the results form torch tensor to numpy
            results_np = np.squeeze(results.detach().numpy())

            if (
                not self.problem_type == "fully_categorical"
                and not self.has_descriptors
            ):
                # reverse transform the inputs
                results_np = reverse_normalize(
                    results_np, self._mins_x, self._maxs_x
                )

            if choices_feat is not None:
                choices_feat = reverse_normalize(
                    choices_feat, self._mins_x, self._maxs_x
                )

            # project the sample back to Olympus format
            sample = project_to_olymp(
                results_np,
                self.param_space,
                has_descriptors=self.has_descriptors,
                choices_feat=choices_feat,
                choices_cat=choices_cat,
            )
            return_params = ParameterVector().from_dict(
                sample, self.param_space
            )

        return return_params


# -----------
# DEBUGGING
# -----------
if __name__ == "__main__":
    PARAM_TYPE = "mixed"

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

        planner = Botorch(goal="minimize")
        planner.set_param_space(param_space)

        campaign = Campaign()
        campaign.set_param_space(param_space)

        BUDGET = 24

        for num_iter in range(BUDGET):

            samples = planner.recommend(campaign.observations)
            print(f"ITER : {num_iter}\tSAMPLES : {samples}")
            # for sample in samples:
            sample_arr = samples.to_array()
            measurement = surface(sample_arr.reshape((1, sample_arr.shape[0])))
            campaign.add_observation(sample_arr, measurement[0])

    elif PARAM_TYPE == "categorical":

        surface_kind = "CatDejong"
        surface = Surface(kind=surface_kind, param_dim=2, num_opts=21)

        campaign = Campaign()
        campaign.set_param_space(surface.param_space)

        planner = Botorch(goal="minimize")
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

        planner = Botorch(goal="minimize")
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
