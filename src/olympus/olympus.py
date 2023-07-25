#!/usr/bin/env python

# ===============================================================================

from . import Logger, __home__, __scratch__
from .campaigns import Campaign
from .databases import Database
from .datasets import Dataset
from .emulators import Emulator
from .evaluators import Evaluator
from .objects import Object
from .planners import (
    Planner,
    get_cat_planners_list,
    get_cont_planners_list,
    get_disc_planners_list,
    get_planners_list,
)
from .plotter import Plotter
from .scalarizers import Scalarizer
from .surfaces import (
    Surface,
    list_cat_surfaces,
    list_cont_surfaces,
    list_surfaces,
)

# ===============================================================================


class Olympus(Object):

    """Master class of the olympus package"""

    def __init__(self, *args, **kwargs):
        Object.__init__(**locals())
        self.home = __home__
        self.scratch = __scratch__
        self.database = Database()

    def _check_planner_param_type(self, planner, param_type):
        map_ = {
            "continuous": get_cont_planners_list,
            "discrete": get_disc_planners_list,
            "categorical": get_cat_planners_list,
        }
        return planner in map_[param_type]()

    # *** Production ****************************************************************
    def run(
        self,
        planner="Phoenics",
        dataset="alkox",
        model="BayesNeuralNet",
        goal="default",
        campaign=Campaign(),
        database=Database(),
        num_iter=3,
    ):
        # check the dataset type
        # TODO: can we check this without creating the object here??
        dataset_obj = Dataset(kind=dataset)

        for param_type in dataset_obj.param_types:
            pass
            # if not self._check_planner_param_type(planner, param_type):
            #     message = f'Planner {planner} cannot handle {param_type} parameters!'
            #     Logger.log(message, 'FATAL')

        if "continuous" in dataset_obj.param_types:
            # we need a NN emulator
            emulator = Emulator(dataset=dataset, model=model)
        else:
            # fully categorical and/or discrete, we use the dataset object as
            # a lookup table in place of the NN emulator, handled in Evaluator
            emulator = dataset_obj

        if goal == "default":
            goal = dataset_obj.goal

        planner_ = Planner(kind=planner, goal=goal)
        # set links
        self.planner = planner_
        self.emulator = emulator
        self.campaign = campaign
        self.database = database

        # define evaluator and optimize
        self.evaluator = Evaluator(
            planner=planner_,
            emulator=emulator,
            campaign=campaign,
            database=database,
        )
        self.evaluator.optimize(num_iter=num_iter)

    def run_analytic(
        self,
        planner="Phoenics",
        surface="Dejong",
        param_dim=2,
        num_opts=None,
        goal="minimize",
        scalarizer=None,
        campaign=Campaign(),
        database=Database(),
        num_iter=3,
    ):

        self.planner = Planner(kind=planner, goal=goal)
        # check if surface is categorical, and check planner
        # param type compatibility
        if surface in list_cont_surfaces():
            if not "continuous" in self.planner.PARAM_TYPES:
                message = (
                    f"Planner {planner} does not support continuous parameters"
                )
                Logger.log(message, "FATAL")
        elif surface in list_cat_surfaces():
            if not "categorical" in self.planner.PARAM_TYPES:
                message = f"Planner {planner} does not support categorical parameters"
                Logger.log(message, "FATAL")

        self.surface = Surface(
            kind=surface, param_dim=param_dim, num_opts=num_opts
        )

        if len(self.surface.value_space) > 1 and not scalarizer:
            message = f"You must pass a scalarizer instance for multiobjective optimization in Olympus"
            Logger.log(message, "FATAL")
        elif len(self.surface.value_space) > 1 and scalarizer:
            self.scalarizer = scalarizer
        else:
            self.scalarizer = None

        self.campaign = campaign
        self.database = database
        self.evaluator = Evaluator(
            planner=self.planner,
            emulator=self.surface,
            campaign=self.campaign,
            scalarizer=self.scalarizer,
            database=self.database,
        )
        self.evaluator.optimize(num_iter=num_iter)

    def benchmark(
        self,
        dataset="alkox",
        planners="all",
        database=Database(kind="sqlite"),
        num_ind_runs=5,
        num_iter=3,
    ):
        """
        Args:
            dataset (str): the dataset to use
        """
        if planners == "all":
            planners = get_planners_list()

        for planner in planners:
            for _ in range(num_ind_runs):
                self.run(
                    planner=planner,
                    dataset=dataset,
                    database=database,
                    campaign=Campaign(),
                    num_iter=num_iter,
                )


# *** Analysis ******************************************************************

# 	def load_database(self, file_name):
# 		''' connects to a database previously written by olympus
#
# 		Args:
# 			file_name (str or list): path and name of the database file
# 		'''
# 		if hasattr(self, 'database'):
# 			self.database.from_file(file_name)
# 		else:
# 			self.database = Database().from_file(file_name)
#
#
# 	def get_campaigns(self, file_names=[]):
# 		if len(file_names) > 0:
# 			return Database().from_file(file_names).get_campaigns()
# 		else:
# 			return self.database.get_campaigns()

# ===============================================================================
