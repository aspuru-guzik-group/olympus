#!/usr/bin/env python

# ===============================================================================

from .campaigns import Campaign
from .databases import Database
from .emulators import Emulator
from .evaluators import Evaluator
from .objects import Object
from .plotter import Plotter
from .planners import Planner, get_planners_list
from .surfaces import Surface

from . import __home__, __scratch__

# ===============================================================================


class Olympus(Object):

    """ Master class of the olympus package
    """

    def __init__(self, *args, **kwargs):
        Object.__init__(**locals())
        self.home = __home__
        self.scratch = __scratch__
        self.database = Database()

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

        emulator = Emulator(dataset=dataset, model=model)
        if goal == "default":
            goal = emulator.goal
        planner_ = Planner(kind=planner, goal=goal)

        # set links
        self.planner = planner_
        self.emulator = emulator
        self.campaign = campaign
        self.database = database

        # define evaluator and optimize
        self.evaluator = Evaluator(
            planner=planner_, emulator=emulator, campaign=campaign, database=database,
        )
        self.evaluator.optimize(num_iter=num_iter)

    def run_analytic(
        self,
        planner="Phoenics",
        surface="dejong",
        param_dim=2,
        goal="minimize",
        campaign=Campaign(),
        database=Database(),
        num_iter=3,
    ):

        self.planner = Planner(kind=planner, goal=goal)
        self.surface = Surface(name=surface, param_dim=param_dim)
        self.campaign = campaign
        self.database = database
        self.evaluator = Evaluator(
            planner=self.planner,
            emulator=self.surface,
            campaign=self.campaign,
            database=self.database,
        )
        self.evaluator.optimize(num_iter=num_iter)

    def benchmark(
        self,
        dataset="alkox",
        planners="all",
        database=Database(kind="sqlite"),
        num_ind_runs=5,
        num_iter=3
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
                    num_iter=num_iter
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
