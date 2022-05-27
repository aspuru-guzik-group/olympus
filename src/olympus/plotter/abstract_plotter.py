#!/usr/bin/env python


import abc
import itertools

import numpy as np

from olympus import Logger, Object
from olympus.datasets import Dataset
from olympus.emulators import Emulator
from olympus.surfaces import Surface


class AbstractPlotter(Object):
    @abc.abstractmethod
    def _plot_traces(
        self,
        emulators,
        planners,
        measurements,
        plot_file_name,
        *args,
        **kwargs,
    ):
        pass

    @abc.abstractmethod
    def _plot_traces_regret(
        self,
        emulators,
        planners,
        measurements,
        plot_file_name,
        *args,
        **kwargs,
    ):
        pass

    @abc.abstractmethod
    def _plot_traces_rank(
        self,
        emulators,
        planners,
        measurements,
        plot_file_name,
        *args,
        **kwargs,
    ):
        pass

    @abc.abstractmethod
    def _plot_traces_fraction_top_k(
        self,
        emulators,
        planners,
        measurements,
        threshold,
        is_percent,
        plot_file_name,
        *args,
        **kwargs,
    ):
        pass

    @abc.abstractmethod
    def _plot_num_evals_top_k(
        self,
        emulators,
        planners,
        measurements,
        threshold,
        is_percent,
        plot_file_name,
        *args,
        **kwargs,
    ):
        pass

    @abc.abstractmethod
    def _plot_regret_x_evals(
        self,
        emulators,
        planners,
        measurements,
        num_evals,
        is_cumulative,
        plot_file_name,
        *args,
        **kwargs,
    ):
        pass

    def plot_from_db(
        self,
        database,
        kind="traces",
        emulator=None,
        plot_file_name="test.png",
        threshold=None,
        is_percent=False,
        num_evals=None,
        *args,
        **kwargs,
    ):
        # unpack the campaigns from the database
        campaigns = database.get_campaigns()
        emulators = []
        planners = []
        measurements = {}

        problem_types = []

        # determine problem type
        for campaign in campaigns:
            # unpack parameter/value types, infer the problem type
            problem_types.append(
                self._infer_problem_type(
                    campaign.param_space,
                    campaign.value_space,
                )
            )

        # check to see if we have all the same problem types throughout the campaings
        problem_type = list(set(problem_types))
        if len(problem_type) > 1:
            message = f'Your database must include campaigns with the same probelm type. You have provided types : {", ".join(problem_type)}'
            Logger.log("FATAL")

        self._validate_plot_kind(problem_type, kind)

        # delegate parsing based on requested plot kind
        if kind == "traces":
            emulators, planners, measurements = self._get_traces(campaigns)
            self._plot_traces(
                emulators,
                planners,
                measurements,
                plot_file_name,
                *args,
                **kwargs,
            )
        elif kind == "traces_regret":
            emulators, planners, measurements = self._get_traces_regret(
                campaigns
            )
            self._plot_traces_regret(
                emulators,
                planners,
                measurements,
                plot_file_name,
                *args,
                **kwargs,
            )
        elif kind == "traces_rank":
            emulators, planners, measurements = self._get_traces_rank(
                campaigns
            )
            self._plot_traces_rank(
                emulators,
                planners,
                measurements,
                plot_file_name,
                *args,
                **kwargs,
            )
        elif kind == "traces_fraction_top_k":
            (
                emulators,
                planners,
                measurements,
            ) = self._get_traces_fraction_top_k(
                campaigns, threshold, is_percent
            )
            self._plot_traces_fraction_top_k(
                emulators,
                planners,
                measurements,
                threshold,
                is_percent,
                plot_file_name,
                *args,
                **kwargs,
            )
        elif kind == "num_evals_top_k":
            emulators, planners, measurements = self._get_num_evals_top_k(
                campaigns, threshold, is_percent
            )
            self._plot_num_evals_top_k(
                emulators,
                planners,
                measurements,
                threshold,
                is_percent,
                plot_file_name,
                *args,
                **kwargs,
            )
        elif kind == "cumulative_regret_x_evals":
            is_cumulative = True
            emulators, planners, measurements = self._get_regret_x_evals(
                campaigns, num_evals, is_cumulative=is_cumulative
            )
            self._plot_regret_x_evals(
                emulators,
                planners,
                measurements,
                num_evals,
                is_cumulative,
                plot_file_name,
                *args,
                **kwargs,
            )
        elif kind == "regret_x_evals":
            is_cumulative = False
            emulators, planners, measurements = self._get_regret_x_evals(
                campaigns, num_evals, is_cumulative=is_cumulative
            )
            self._plot_regret_x_evals(
                emulators,
                planners,
                measurements,
                num_evals,
                is_cumulative,
                plot_file_name,
                *args,
                **kwargs,
            )
        else:
            raise NotImplementedError

    # parse data for trace-type plots
    def _get_traces(self, campaigns):

        emulators = []
        planners = []
        measurements = {}

        for campaign in campaigns:
            # parse emulator type
            if (
                campaign.emulator_type == "n/a"
            ):  # this means campaign never got an Emulator or Surface
                continue
            emulator_type = campaign.get_emulator_type()
            if emulator_type == "numeric":
                # neural network based emulator or lookup table for fully categorical
                camp_emulator = (
                    f"{campaign.model_kind}_{campaign.dataset_kind}"
                )
            elif emulator_type == "analytic":
                # analytic surface
                camp_emulator = f"analytic_{campaign.surface_kind}"
            # if emulator is None:
            emulator = camp_emulator
            if camp_emulator != emulator:
                continue

            # parse planner information
            planner = campaign.get_planner_kind()
            if emulator not in emulators:
                emulators.append(emulator)
                measurements[emulator] = {}
            if planner not in planners:
                planners.append(planner)
                measurements[emulator][planner] = {"idxs": [], "vals": []}
            for val_index, value in enumerate(campaign.best_values):
                measurements[emulator][planner]["idxs"].append(val_index)
                measurements[emulator][planner]["vals"].append(value)

        for emulator in emulators:
            for planner in planners:
                try:
                    measurements[emulator][planner]["idxs"] = np.array(
                        measurements[emulator][planner]["idxs"]
                    )
                    measurements[emulator][planner]["vals"] = np.array(
                        measurements[emulator][planner]["vals"]
                    )
                except KeyError:
                    continue
            pass

        return emulators, planners, measurements

    def _get_traces_regret(self, campaigns):

        emulators = []
        planners = []
        measurements = {}

        for campaign in campaigns:
            # get the global optimum objective value for the campaign
            goal = campaign.goal

            # parse emulator type
            if (
                campaign.emulator_type == "n/a"
            ):  # this means campaign never got an Emulator or Surface
                continue
            emulator_type = campaign.get_emulator_type()
            if emulator_type == "numeric":
                # neural network based emulator or lookup table for fully categorical
                camp_emulator = (
                    f"{campaign.model_kind}_{campaign.dataset_kind}"
                )
                # TODO: implement estimate of best objective value for the
                # emulated datasets here
            elif emulator_type == "analytic":
                # analytic surface
                camp_emulator = f"analytic_{campaign.surface_kind}"
                surf = Surface(
                    kind=campaign.surface_kind,
                    param_dim=len(campaign.param_space),
                )  # continuous surfaces only
                if goal == "minimize":
                    opt_obj = surf.minima[0]["value"]
                elif goal == "maximize":
                    opt_obj = surf.maxima[0]["value"]

            # parse planner information
            planner = campaign.get_planner_kind()
            if camp_emulator not in emulators:
                emulators.append(camp_emulator)
                measurements[camp_emulator] = {}
            if planner not in planners:
                planners.append(planner)
                measurements[camp_emulator][planner] = {"idxs": [], "vals": []}
            for val_index, value in enumerate(campaign.best_values):
                measurements[camp_emulator][planner]["idxs"].append(val_index)
                measurements[camp_emulator][planner]["vals"].append(
                    np.abs(value - opt_obj)
                )  # regret

        for emulator in emulators:
            for planner in planners:
                try:
                    measurements[emulator][planner]["idxs"] = np.array(
                        measurements[emulator][planner]["idxs"]
                    )
                    measurements[emulator][planner]["vals"] = np.array(
                        measurements[emulator][planner]["vals"]
                    )
                except KeyError:
                    continue
            pass

        return emulators, planners, measurements

    def _get_traces_rank(self, campaigns):
        emulators = []
        planners = []
        measurements = {}

        for campaign in campaigns:
            # parse emulator type
            if (
                campaign.emulator_type == "n/a"
            ):  # this means campaign never got an Emulator or Surface
                continue
            emulator_type = campaign.get_emulator_type()
            if emulator_type == "numeric":
                # lookup table for fully categorical
                camp_emulator = (
                    f"{campaign.model_kind}_{campaign.dataset_kind}"
                )

                dataset = Dataset(kind=campaign.dataset_kind)
                values = dataset.targets.values
                sort_values = np.sort(values, axis=0)
                if campaign.goal == "minimize":
                    pass
                elif campaign.goal == "maximize":
                    sort_values = sort_values[::-1]
                ranks = {v[0]: i + 1 for i, v in enumerate(sort_values)}

            elif emulator_type == "analytic":
                # analytic surface
                camp_emulator = f"analytic_{campaign.surface_kind}"

                surf = Surface(
                    kind=campaign.surface_kind,
                    param_dim=len(campaign.param_space),
                    num_opts=len(
                        campaign.param_space[0]["options"]
                    ),  # num_opts will be the same along each dimension
                )
                values = np.array(
                    surf.run(self._create_available_options(surf.param_space))
                ).squeeze()
                sort_values = np.sort(values, axis=0)
                if campaign.goal == "minimize":
                    pass
                elif campaign.goal == "maximize":
                    sort_values = sort_values[::-1]
                ranks = {v: i + 1 for i, v in enumerate(sort_values)}

            # parse planner information
            planner = campaign.get_planner_kind()
            if camp_emulator not in emulators:
                emulators.append(camp_emulator)
                measurements[camp_emulator] = {}
            if planner not in planners:
                planners.append(planner)
                measurements[camp_emulator][planner] = {"idxs": [], "vals": []}
            for val_index, value in enumerate(campaign.best_values):
                measurements[camp_emulator][planner]["idxs"].append(val_index)
                measurements[camp_emulator][planner]["vals"].append(
                    ranks[value[0]]
                )  # rank

        # TODO: what is this for??
        for emulator in emulators:
            for planner in planners:
                try:
                    measurements[emulator][planner]["idxs"] = np.array(
                        measurements[emulator][planner]["idxs"]
                    )
                    measurements[emulator][planner]["vals"] = np.array(
                        measurements[emulator][planner]["vals"]
                    )
                except KeyError:
                    continue
            pass

        return emulators, planners, measurements

    def _get_traces_moo():
        pass

    def _get_traces_fraction_top_k(
        self, campaigns, threshold, is_percent=False
    ):

        # quickly validate the threshold argument
        if not (isinstance(threshold, float) or isinstance(threshold, int)):
            message = "threshold argument must be of type float or int"
            Logger.log(message, "FATAL")
        if is_percent:
            if not (0.0 <= threshold <= 100.0):
                message = "for percentages, threshold value must be between 0 and 100"
                Logger.log(message, "FATAL")
        else:
            # this number is now intepreted as a `number` of candidates
            pass

        emulators = []
        planners = []
        measurements = {}

        for campaign in campaigns:
            # parse emulator type
            if (
                campaign.emulator_type == "n/a"
            ):  # this means campaign never got an Emulator or Surface
                continue
            emulator_type = campaign.get_emulator_type()

            if emulator_type == "numeric":
                # neural network based emulator or lookup table for fully categorical
                camp_emulator = (
                    f"{campaign.model_kind}_{campaign.dataset_kind}"
                )

                dataset = Dataset(kind=campaign.dataset_kind)
                values = dataset.targets.values

                sort_idxs = np.argsort(values, axis=0)
                if campaign.goal == "minimize":
                    pass
                elif campaign.goal == "maximize":
                    sort_idxs = sort_idxs[::-1]

                sort_values = [values[i] for i in sort_idxs]

                if not is_percent:
                    sort_values = sort_values[:threshold]
                else:
                    percent_val = (threshold / 100) * (
                        np.amax(sort_values) - np.amin(sort_values)
                    ) + np.amin(sort_values)
                    if campaign.goal == "minimize":
                        sort_values = sort_values[
                            : np.amin(np.where(sort_values > percent_val))
                        ]
                    if campaign.goal == "maximize":
                        sort_values = sort_values[
                            : np.amin(np.where(sort_value < percent_val))
                        ]

            elif emulator_type == "analytic":
                # analytic surface
                camp_emulator = f"analytic_{campaign.surface_kind}"

                surf = Surface(
                    kind=campaign.surface_kind,
                    param_dim=len(campaign.param_space),
                    num_opts=len(
                        campaign.param_space[0]["options"]
                    ),  # num_opts will be the same along each dimension
                )
                # params = np.array(self._create_available_options(surf.param_space)).squeeze()
                values = np.array(
                    surf.run(self._create_available_options(surf.param_space))
                ).squeeze()
                sort_idxs = np.argsort(values, axis=0)
                if campaign.goal == "minimize":
                    pass
                elif campaign.goal == "maximize":
                    sort_idxs = sort_idxs[::-1]

                sort_values = [values[i] for i in sort_idxs]

                if not is_percent:
                    sort_values = sort_values[:threshold]
                else:
                    percent_val = (threshold / 100) * (
                        np.amax(sort_values) - np.amin(sort_values)
                    ) + np.amin(sort_values)
                    if campaign.goal == "minimize":
                        sort_values = sort_values[
                            : np.amin(np.where(sort_values > percent_val))
                        ]
                    if campaign.goal == "maximize":
                        sort_values = sort_values[
                            : np.amin(np.where(sort_value < percent_val))
                        ]

            # parse planner information
            planner = campaign.get_planner_kind()
            if camp_emulator not in emulators:
                emulators.append(camp_emulator)
                measurements[camp_emulator] = {}
            if planner not in planners:
                planners.append(planner)
                measurements[camp_emulator][planner] = {"idxs": [], "vals": []}
            # for val_index, value in enumerate(campaign.best_values):
            #    measurements[camp_emulator][planner]['idxs'].append(val_index)
            #    measurements[camp_emulator][planner]['vals'].append(value)
            campaign_values = campaign.get_values()
            for val_index in range(len(campaign_values)):
                measurements[camp_emulator][planner]["idxs"].append(val_index)
                measurements[camp_emulator][planner]["vals"].append(
                    sum(
                        val in campaign_values[:val_index]
                        for val in sort_values
                    )
                    / len(sort_values)
                )  # fraction of top candidates

        for emulator in emulators:
            for planner in planners:
                try:
                    measurements[emulator][planner]["idxs"] = np.array(
                        measurements[emulator][planner]["idxs"]
                    )
                    measurements[emulator][planner]["vals"] = np.array(
                        measurements[emulator][planner]["vals"]
                    )
                except KeyError:
                    continue
            pass

        return emulators, planners, measurements

    # parse data for box/violin-type plots
    def _get_num_evals_top_k(self, campaigns, threshold, is_percent=False):

        # quickly validate the threshold argument
        if not (isinstance(threshold, float) or isinstance(threshold, int)):
            message = "threshold argument must be of type float or int"
            Logger.log(message, "FATAL")
        if is_percent:
            if not (0.0 <= threshold <= 100.0):
                message = "for percentages, threshold value must be between 0 and 100"
                Logger.log(message, "FATAL")
        else:
            # this number is now intepreted as a `number` of candidates
            pass

        emulators = []
        planners = []
        measurements = {}

        for campaign in campaigns:
            # parse emulator type
            if (
                campaign.emulator_type == "n/a"
            ):  # this means campaign never got an Emulator or Surface
                continue
            emulator_type = campaign.get_emulator_type()

            if emulator_type == "numeric":
                # lookup table for fully categorical
                camp_emulator = (
                    f"{campaign.model_kind}_{campaign.dataset_kind}"
                )

                dataset = Dataset(kind=campaign.dataset_kind)
                values = dataset.targets.values

                sort_idxs = np.argsort(values, axis=0)
                if campaign.goal == "minimize":
                    pass
                elif campaign.goal == "maximize":
                    sort_idxs = sort_idxs[::-1]

                sort_values = [values[i] for i in sort_idxs]

                if not is_percent:
                    sort_values = sort_values[:threshold]
                else:
                    percent_val = (threshold / 100) * (
                        np.amax(sort_values) - np.amin(sort_values)
                    ) + np.amin(sort_values)
                    if campaign.goal == "minimize":
                        sort_values = sort_values[
                            : np.amin(np.where(sort_values > percent_val))
                        ]
                    if campaign.goal == "maximize":
                        sort_values = sort_values[
                            : np.amin(np.where(sort_value < percent_val))
                        ]

            elif emulator_type == "analytic":
                # analytic surface
                camp_emulator = f"analytic_{campaign.surface_kind}"

                surf = Surface(
                    kind=campaign.surface_kind,
                    param_dim=len(campaign.param_space),
                    num_opts=len(
                        campaign.param_space[0]["options"]
                    ),  # num_opts will be the same along each dimension
                )
                # params = np.array(self._create_available_options(surf.param_space)).squeeze()
                values = np.array(
                    surf.run(self._create_available_options(surf.param_space))
                ).squeeze()
                sort_idxs = np.argsort(values, axis=0)
                if campaign.goal == "minimize":
                    pass
                elif campaign.goal == "maximize":
                    sort_idxs = sort_idxs[::-1]

                sort_values = [values[i] for i in sort_idxs]

                if not is_percent:
                    sort_values = sort_values[:threshold]
                else:
                    percent_val = (threshold / 100) * (
                        np.amax(sort_values) - np.amin(sort_values)
                    ) + np.amin(sort_values)
                    if campaign.goal == "minimize":
                        sort_values = sort_values[
                            : np.amin(np.where(sort_values > percent_val))
                        ]
                    if campaign.goal == "maximize":
                        sort_values = sort_values[
                            : np.amin(np.where(sort_value < percent_val))
                        ]

            # parse planner information
            planner = campaign.get_planner_kind()
            if camp_emulator not in emulators:
                emulators.append(camp_emulator)
                measurements[camp_emulator] = {}
            if planner not in planners:
                planners.append(planner)
                measurements[camp_emulator][planner] = {
                    "planner": [],
                    "vals": [],
                }

            campaign_values = campaign.get_values()
            num_evals = 1
            for val_index, val in enumerate(campaign_values):
                if val in sort_values:
                    break
                else:
                    pass
                num_evals += 1
            measurements[camp_emulator][planner]["vals"].append(num_evals)
            measurements[camp_emulator][planner]["planner"].append(planner)

        for emulator in emulators:
            for planner in planners:
                try:
                    # measurements[emulator][planner]['planner'] = np.array(measurements[emulator][planner]['idxs'])
                    measurements[emulator][planner]["vals"] = np.array(
                        measurements[emulator][planner]["vals"]
                    )
                except KeyError:
                    continue
            pass

        return emulators, planners, measurements

    def _get_regret_x_evals(self, campaigns, num_evals, is_cumulative=False):

        emulators = []
        planners = []
        measurements = {}

        for campaign in campaigns:
            # get the global optimum objective value for the campaign
            goal = campaign.goal

            # parse emulator type
            if (
                campaign.emulator_type == "n/a"
            ):  # this means campaign never got an Emulator or Surface
                continue
            emulator_type = campaign.get_emulator_type()
            if emulator_type == "numeric":
                # neural network based emulator or lookup table for fully categorical
                camp_emulator = (
                    f"{campaign.model_kind}_{campaign.dataset_kind}"
                )
                # TODO: implement estimate of best objective value for the
                # emulated datasets here
            elif emulator_type == "analytic":
                # analytic surface
                camp_emulator = f"analytic_{campaign.surface_kind}"
                surf = Surface(
                    kind=campaign.surface_kind,
                    param_dim=len(campaign.param_space),
                )  # continuous surfaces only
                if goal == "minimize":
                    opt_obj = surf.minima[0]["value"]
                elif goal == "maximize":
                    opt_obj = surf.maxima[0]["value"]

            # parse planner information
            planner = campaign.get_planner_kind()
            if camp_emulator not in emulators:
                emulators.append(camp_emulator)
                measurements[camp_emulator] = {}
            if planner not in planners:
                planners.append(planner)
                measurements[camp_emulator][planner] = {
                    "planner": [],
                    "vals": [],
                }

            if is_cumulative:
                regret_to_add = np.sum(
                    campaign.best_values[:num_evals] - opt_obj
                )  # cumulative regret
            else:
                regret_to_add = (
                    campaign.best_values[num_evals] - opt_obj
                )  # instantaneous regret
            measurements[camp_emulator][planner]["vals"].append(regret_to_add)
            measurements[camp_emulator][planner]["planner"].append(planner)

        for emulator in emulators:
            for planner in planners:
                try:
                    # measurements[emulator][planner]['idxs'] = np.array(measurements[emulator][planner]['idxs'])
                    measurements[emulator][planner]["vals"] = np.array(
                        measurements[emulator][planner]["vals"]
                    )
                except KeyError:
                    continue
            pass

        return emulators, planners, measurements

    def _validate_plot_kind(self, problem_type, kind):
        # check to see if the kind of plot is within the supported plot kinds
        supported_kinds = [
            method.split("_plot_")[1]
            for method in dir(self)
            if method.startswith("_plot_")
        ]
        if not kind in supported_kinds:
            message = f'Olympus does not suppot plot kind {kind}. Please choose from one of : {", ".join(supported_kinds)}'
            Logger.log(message, "FATAL")

        # TODO: check if the problem type macthes with the kind

    def _infer_problem_type(self, param_space, value_space):
        """infer the parameter space from Olympus. The three possibilities are
        "fully_continuous", "mixed", or "fully_categorical"

        Args:
            param_space (obj): Olympus parameter space object
        """
        param_types = [p.type for p in param_space]
        if param_types.count("continuous") == len(param_types):
            problem_type = "fully_continuous"
        elif param_types.count("categorical") == len(param_types):
            problem_type = "fully_categorical"
        elif np.logical_and(
            "continuous" in param_types, "categorical" in param_types
        ):
            problem_type = "mixed"
        if len(value_space) > 1:
            problem_type = "-".join([problem_type, "moo"])
        return problem_type

    def _create_available_options(self, param_space):
        """build cartesian product space of options, then remove options
        which have already been measured. Returns an (num_options, num_dims)
        torch tensor with all possible options

        Args:
            param_space (obj): Olympus parameter space object
        """
        param_names = [p.name for p in param_space]
        param_options = [p.options for p in param_space]

        cart_product = list(itertools.product(*param_options))
        cart_product = [list(elem) for elem in cart_product]

        # current_avail_feat  = []
        current_avail_cat = []
        for elem in cart_product:
            current_avail_cat.append(elem)

        return current_avail_cat
