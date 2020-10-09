#!/usr/bin/env python

import numpy as np
from abc import abstractmethod
from olympus import Logger
from olympus.campaigns import Observations, Campaign
from olympus.objects import Object, abstract_attribute, ABCMeta, Config


class AbstractPlanner(Object, metaclass=ABCMeta):
    """ This class is intended to contain methods shared by all wrappers, as well as define abstract methods and
    attributes that all planner wrappers need to implement. It is not meant to be exposed to the standard user,
    although more advanced users can use this class to create custom planner classes.
    """

    def __init__(self, *args, **kwargs):
        Object.__init__(self, *args, **kwargs)
        self.num_generated = 0
        self.param_space = None
        self._params = None
        self._values = None
        self.SUBMITTED_PARAMS = []
        self.RECEIVED_VALUES  = []

        # rm all those vars in config that are not needed/used by ScipyWrapper
        for var in ['goal', 'init_guess', 'random_seed']:
            if var in kwargs:
                del kwargs[var]
        self.config = Config(from_dict=kwargs)

        # self.goal is an abstract attribute that needs to be defined by the subclasses of AbstractPlanner
        # Since all planner wrappers are implemented in minimization mode, we flip the measurements if we want to
        # perform a maximization
        if self.goal == 'minimize':
            self.flip_measurements = False
        elif self.goal == 'maximize':
            self.flip_measurements = True
        else:
            message = f'attribute `goal` can only be "minimize" or "maximize". "{self.goal}" is not a valid value'
            Logger.log(message, 'ERROR')

    def __repr__(self):
        if not hasattr(self, 'param_space') or self.param_space is None:
            param_space = 'undefined'
        else:
            param_space = 'defined'
        return f"<Planner (kind={self.kind}, param_space={param_space})>"

    @abstract_attribute
    def goal(self):
        pass

    @property
    def kind(self):
        return type(self).__name__

    @abstractmethod
    def _set_param_space(self, *args, **kwargs):
        """Method that returns the parameter space over which to optimize in the format used by a specific optimization
        library/wrapper.
        """
        pass

    @abstractmethod
    def _tell(self, *args, **kwargs):
        """Method that returns all previous observations as required by a specific optimization library/wrapper.
        """
        pass

    @abstractmethod
    def _ask(self, *args, **kwargs):
        """Method that returns the next query point based on all previous observations.
        """
        pass

    def set_param_space(self, param_space):
        """Defines the parameter space over which the planner will search.

        Args:
            param_space (ParameterSpace): a ParameterSpace object defining the space over which to search.
        """
        self.param_space = param_space
        self._set_param_space(param_space)

    def ask(self, return_as=None):
        """ suggest new set of parameters

        Args:
            return_as (string): choose data type for returned parameters
                allowed options (dict, array)

        Returns:
            ParameterVector: newly generated parameters
        """

        self.num_generated += 1
        param_vector = self._ask()

        # check that the parameters suggested are within the bounds of our param_space
        self._validate_paramvector(param_vector)

        if return_as is not None:
            try:
                param_vector = getattr(param_vector, 'to_{}'.format(return_as))()
            except AttributeError:
                Logger.log('could not return param_vector as "{}"'.format(return_as), 'ERROR')
        return param_vector

    def tell(self, observations=Observations()):
        """Provide the planner with all previous observations.

        Args:
            observations (Observations): an Observation object containing all previous observations. This defines the
                history of the campaign seen by the planner. The default is None, i.e. there are no previous
                observations.
        """
        self._tell(observations)

    def recommend(self, observations=None, return_as=None):
        """Consecutively executes tell and ask: tell the planner about all previous observations, and ask about the
        next query point.

        Args:
            observations (list of ???)
            return_as (string): choose data type for returned parameters
                allowed options (dict, array)

        Returns:
            list: newly generated parameters
        """
        self.tell(observations)
        return self.ask(return_as=return_as)

    def optimize(self, emulator, num_iter=1, verbose=False):
        """Optimizes a surface for a fixed number of iterations.

        Args:
            emulator (object): Emulator or a Surface instance to optimize over.
            num_iter (int): Maximum number of iterations allowed.
            verbose (bool): Whether to print information to screen.

        Returns:
            campaign (Campaign): Campaign object with information about the optimization, including all parameters
                tested and measurements obtained.
        """

        # update num_iter if needed by the specific wrapper
        if hasattr(self, 'num_iter') and self.num_iter != num_iter:
            Logger.log(f'Updating the number of sampling points of planner {type(self).__name__} to {num_iter}', 'INFO')
            self.num_iter = num_iter

        # same for budget
        if hasattr(self, 'budget') and self.budget != num_iter:
            Logger.log(f'Updating the number of sampling points of planner {type(self).__name__} to {num_iter}', 'INFO')
            self.budget = num_iter

        # reset planner if it has a 'reset' method. Assuming that if there is a 'reset' method it is needed here
        # This is used by Deap for example, to clear the latest population before doing another optimization
        if callable(getattr(self, "reset", None)):
            self.reset()

        # use campaign to store info, and then to be returned
        campaign = Campaign()
        campaign.set_planner_specs(self)
        campaign.set_emulator_specs(emulator)

        # provide the planner with the parameter space.
        # param space in emulator as it originates from dataset
        self.set_param_space(emulator.param_space)

        # Optimize: i.e. call the planner recommend method for max_iter times
        for i in range(num_iter):
            if verbose is True:
                Logger.log(f'Optimize iteration {i+1}', 'INFO')
                Logger.log(f'Obtaining parameters from planner...', 'INFO')
            # get new params from planner
            # NOTE: now we get 1 param at a time, a possible future expansion is to return batches
            params = self.recommend(observations=campaign.observations)

            # get measurement from emulator/surface
            if verbose is True:
                Logger.log(f'Obtaining measurement from emulator...', 'INFO')
            values = emulator.run(params.to_array(), return_paramvector=True)

            # store parameter and measurement pair in campaign
            campaign.add_observation(params, values)

        return campaign

    def _validate_paramvector(self, param_vector):
        for key, value in param_vector.to_dict().items():
            param = self.param_space.get_param(name=key)
            if param['type'] == 'continuous':
                if not param['low'] <= value <= param['high']:
                    message = 'Proposed parameter {0} not within defined bounds ({1},{2})'.format(value, param['low'], param['high'])
                    Logger.log(message, 'WARNING')

    def _project_into_domain(self, params):
        for _, bound in enumerate(self.param_space):
            params[_] = np.maximum(bound['low'], params[_])
            params[_] = np.minimum(bound['high'], params[_])
        return params
