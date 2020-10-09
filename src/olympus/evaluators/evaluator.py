#!/usr/bin/env python

from olympus import Emulator, Surface, Logger
from olympus.campaigns import Campaign
from olympus.objects   import Object

#===============================================================================

class Evaluator(Object):

    def __init__(self, planner, emulator=None, surface=None, campaign=Campaign(), database=None):
        """ The Evaluator does higher level operations that Planners and 
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
            database (object): ...
        """
        Object.__init__(**locals())

        if emulator is not None:
            assert surface is None
            self.emulator_type = 'numerical'
        elif surface is not None:
            assert emulator is None 
            self.emulator_type = 'analytic'
            self.emulator = surface
        else:
            Logger.log('One of emulator or surface needs to be provided', 'FATAL')

#        if isinstance(self.emulator, Emulator):
#            self.emulator_type = 'numerical'
#        elif isinstance(self.emulator, Surface):
#            self.emulator_type = 'analytic'

        # provide the planner with the parameter space.
        # NOTE: right now, outside of Evaluator, the param_space for a planner 
        #       needs to be set "manually" by the user
        self.planner.set_param_space(self.emulator.param_space)  # param space in emulator as it originates from dataset

        if self.campaign is not None:
            self.campaign.set_planner_specs(planner)
            self.campaign.set_emulator_specs(emulator)


    def optimize(self, num_iter=1):
        """Optimizes a surface for a fixed number of iterations.

        Args:
            num_iter (int): Maximum number of iterations allowed.
        """

        # Optimize: i.e. call the planner recommend method for max_iter times
        for i in range(num_iter):
            # get new params from planner
            # NOTE: now we get 1 param at a time, a possible future expansion is 
            #       to return batches
            params = self.planner.recommend(observations=self.campaign.observations)

            # get measurement from emulator/surface
            values = self.emulator.run(params.to_array(), return_paramvector=True)

            # store parameter and measurement pair in campaign
            if self.campaign is not None:
                self.campaign.add_observation(params, values)

            # if we have a database, log the campaign status
            if self.database is not None:
                self.database.log_campaign(self.campaign)
