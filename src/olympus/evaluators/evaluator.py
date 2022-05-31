#!/usr/bin/env python

from olympus import Logger
from olympus.emulators import Emulator
from olympus.surfaces import Surface
from olympus.campaigns import Campaign
from olympus.objects import Object, ParameterVector
from olympus.scalarizers import Scalarizer
from olympus.utils.data_transformer import cube_to_simpl

# ===============================================================================


class Evaluator(Object):
    def __init__(
        self,
        planner,
        emulator=None,
        surface=None,
        campaign=Campaign(),
        scalarizer=None,
        database=None,
    ):
        """The Evaluator does higher level operations that Planners and
        Emulators do not do on their own. For instance, communicating parameters
        and measurements to each other, keeping track of them ensuring they
        match, and storing these in a Campaign object. All this can also be done
        by the user using planner, emulator and campaign objects, which might
        allow more customization. However, Evaluator provides a convenient
        higher-level interface for common optimization tasks.

        Args:
            planner (Planner): an instance of a Planner.
            emulator (Emulator): an instance of a trained Emulator.
            surface (Surface): an instance of a Surface
            campaign (Campaign): an instance of a Campaign. By default, a new
                Campaign instance is created. If this is set to None, no campaign
                info will be stored.
            scalarizer (Scalarizer): an instance of a Scalarizer (i.e. achievement
                scalarizing function) used for multiobjective optimization problems
            database (object): ...
        """
        Object.__init__(**locals())

        if emulator is not None:
            assert surface is None
            self.emulator_type = "numerical"
        elif surface is not None:
            assert emulator is None
            self.emulator_type = "analytic"
            self.emulator = surface
        else:
            Logger.log(
                "One of emulator or surface needs to be provided", "FATAL"
            )

        #        if isinstance(self.emulator, Emulator):
        #            self.emulator_type = 'numerical'
        #        elif isinstance(self.emulator, Surface):
        #            self.emulator_type = 'analytic'

        # provide the planner with the parameter space.
        # NOTE: right now, outside of Evaluator, the param_space for a planner
        #       needs to be set "manually" by the user
        if self.emulator.parameter_constriants == "simplex":
            # use auxillary parameter space if we have a simplex constraint on parameters
            self.planner.set_param_space(self.emulator.aux_param_space)
        else:
            self.planner.set_param_space(
                self.emulator.param_space
            )  # param space in emulator as it originates from dataset

        # check for provision of scalarizing function if we have a MOO function
        # TODO: should we default here to a weighted sum with
        if self.campaign.is_moo:

            if self.scalarizer is None:
                message = "You must provided an instance of a Scalarizer for multiobjective optimization in Olympus"
                Logger.log("FATAL")

            # make sure the goal of the planner/campaign is minimization (always minimization
            # for multi-objective problems, the individual optimization goals are specified in the
            # the scalarizer)
            if not self.planner.goal == "minimize":
                message = 'For multiobjective optimization in Olympus, the overall optimization goal must be set to "minimize". Updating now ...'
                Logger.log(message, "WARNING")
                self.planner.goal = "minimize"
                self.campaign.goal = "minimize"

        if self.campaign is not None:
            self.campaign.set_planner_specs(planner)
            self.campaign.set_emulator_specs(emulator)

    def optimize(self, num_iter=1):
        """Optimizes an objective for a fixed number of iterations.

        Args:
            num_iter (int): Maximum number of iterations allowed.
        """

        # Optimize: i.e. call the planner recommend method for max_iter times
        for i in range(num_iter):

            # NOTE: now we get 1 param at a time, a possible future expansion is
            #       to return batches

            if self.emulator.parameter_constriants == "simplex":
                # transform the campaign observations from simplex to cube (for planner)
                self.campaign.observations_to_cube()

            if self.campaign.is_moo:
                planner_observations = self.campaign.scalarized_observations
            else:
                planner_observations = self.campaign.observations

            # get new params from planner
            params = self.planner.recommend(observations=planner_observations)

            if self.emulator.parameter_constriants == "simplex":
                params = cube_to_simpl([params.to_array()])
                params = ParameterVector().from_array(
                    params[0], self.emulator.param_space
                )
                self.campaign.observations_to_simpl()

            # get measurement from emulator/surface
            values = self.emulator.run(params, return_paramvector=True)

            # store parameter and measurement pair in campaign
            # TODO: we probably do not need this check for NoneType Campaign here... consider removing
            if self.campaign is not None:
                if self.campaign.is_moo:
                    self.campaign.add_and_scalarize(
                        params, values, self.scalarizer
                    )
                else:
                    self.campaign.add_observation(params, values)

            # if we have a database, log the campaign status
            if self.database is not None:
                self.database.log_campaign(self.campaign)
