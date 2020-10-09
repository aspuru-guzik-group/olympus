#!/usr/bin/env python

import numpy as np
from deap import base, creator, tools
from olympus.planners import AbstractPlanner
from olympus.objects import ParameterVector
from olympus import Logger
from copy import deepcopy


# ==========
# Main Class
# ==========
class Genetic(AbstractPlanner):

    def __init__(self, goal='minimize', pop_size=10, cx_prob=0.5, mut_prob=0.2, verbose=False,
                 mate_args={'function':tools.cxTwoPoint},
                 mutate_args={'function':tools.mutGaussian, 'mu':0, 'sigma':0.2, 'indpb':0.2},
                 select_args={'function':tools.selTournament, 'tournsize':3}):
        """Evolutionary Algorithm implemented using the DEAP library.

        We initialize the population using a uniform probability across the parameter space. We then evaluate
        one individual of the population at a time. After all individuals have been evaluated, we generate a set of
        offsprings via selection, cross-over and mutation, using the functions chosen in `select_args`, `mate_args`
        and `mutate_args`. Only novel offsprings, having a set of parameters different from those of the parent
        population, are chosen to be evaluated, and are then proposed sequentially. Once all these offsprings have
        been evaluated, a new set is generated following the same procedure.

        Args:
            goal (str): The optimization goal, either 'minimize' or 'maximize'. Default is 'minimize'.
            pop_size (int): Size of the population. Default is 10.
            cx_prob (float): Probability with which two individuals are crossed. Default is 0.5.
            mut_prob (float): Probability for mutating an individual. Default is 0.2.
            verbose (bool): Whether to print out information. Default is True.
            mate_args (dict): Dictionary containing the details of the crossover function to be used. This should
                contain all arguments that you want to provided to Toolbox.register("mate"). For all crossover
                operations available see https://deap.readthedocs.io/en/master/api/tools.html.
            mutate_args (dict): Dictionary containing the details of the mutation function to be used. This should
                contain all arguments that you want to provided to Toolbox.register("mutate"). For all mutation
                operations available see https://deap.readthedocs.io/en/master/api/tools.html.
            select_args (dict): Dictionary containing the details of the selection function to be used. This should
                contain all arguments that you want to provided to Toolbox.register("select"). For all selection
                operations available see https://deap.readthedocs.io/en/master/api/tools.html.
        """

        AbstractPlanner.__init__(**locals())

        # define fitness function = minimize
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # -1 for minimisation, +1 for maximisation
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # get Deap toolbox
        self.toolbox = base.Toolbox()

    def reset(self):
        """Clears the current population so that a new optimization would start from a randomly initialised population
        again.
        """
        if hasattr(self, 'pop'):
            # reset attributes
            self.num_generated = 0
            self.pop = []
            self.latest_pop_size = None
            self.offsprings = []
            self.novel_offsprings = []
            self.offsprings_to_be_evaluated = []
            # unregister individuals
            self.toolbox.unregister("individual")
            self.toolbox.unregister("population")
            # (re)define fitness function
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)

    def _register_deap_operations(self):
        """register mate, mutate and select functions"""

        # --------------
        # mate operation
        # --------------
        self.toolbox.register("mate", **self.mate_args)

        # ----------------
        # mutate operation
        # ----------------
        # NOTE: when using default mutate_args, the sigma value is not scaled depending on the range of the parameter.
        # If desired, the scaled sigma values need to be computed outside of this wrapper and passed to the 'sigma'
        # key as a list.
        self.toolbox.register("mutate", **self.mutate_args)

        # ----------------
        # select operation
        # ----------------
        self.toolbox.register("select", **self.select_args)

        # ----------------------------------
        # decorate mate and mutate operations
        # -----------------------------------
        bounds = [param['domain'] for param in self._param_space]
        self.toolbox.decorate("mate", self.project_bounds(bounds))
        self.toolbox.decorate("mutate", self.project_bounds(bounds))

        # Alternative option to control the bounds:
        # self.toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
        # self.toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0,
        #                       indpb=1.0/len(self._param_space))

    def _set_param_space(self, param_space):
        self._param_space = []
        for param in param_space:
            if param.type == 'continuous':
                param_dict = {'name': param.name, 'type': param.type, 'domain': (param.low, param.high)}
            self._param_space.append(param_dict)

        # register here as they depend on the param_space
        self._register_deap_operations()

    def _tell(self, observations):
        self._params = observations.get_params(as_array=False)
        self._values = observations.get_values(as_array=True, opposite=self.flip_measurements)
        # reshape the array so to have shape (num observations, num objectives)
        # WARNING: In this case we are sticking to 1 objective, multiple objective are not supported!
        # TODO: fix the following reshaping to allow multiple objectives
        self._values = np.reshape(self._values, newshape=(np.shape(self._values)[0], 1))

    def _generate_first_population(self):
        if self.verbose is True:
            Logger.log(f'Creating first population of size {self.pop_size}', 'INFO')

        # Structure initializers
        bounds = [param['domain'] for param in self._param_space]
        self.toolbox.register("individual", self.initIndividual, icls=creator.Individual, bounds=bounds)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.pop = self.toolbox.population(n=self.pop_size)
        self.latest_pop_size = self.pop_size

        # delete creator classes
        del creator.Individual
        del creator.FitnessMin

    def _set_population_fitness(self):

        # assign evaluations to each offspring
        offsprings_fitness = self._values[-self.latest_pop_size:, :]
        assert len(self.novel_offsprings) == len(offsprings_fitness)

        # Note we operate on self.novel_offsprings, but this will also update self.offsprings, which is the population
        # used for selection, cross-over and mutation
        for ind, fit in zip(self.novel_offsprings, offsprings_fitness):
            if not ind.fitness.valid:
                ind.fitness.values = (fit[0],)

        # replace the old population by the offsprings.
        self.pop[:] = self.offsprings

    def _generate_offsprings(self):
        if self.verbose is True:
            Logger.log('Creating new offsprings...', 'INFO')

        # Select the next generation individuals
        self.offsprings = self.toolbox.select(self.pop, len(self.pop))
        # Clone the selected individuals
        self.offsprings = list(map(self.toolbox.clone, self.offsprings))

        # Apply crossover and on the offsprings
        for child1, child2 in zip(self.offsprings[::2], self.offsprings[1::2]):
            if np.random.uniform() < self.cx_prob:
                if self.verbose is True:
                    Logger.log('  Performing cross-over operation', 'INFO')
                self.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation on the offsprings
        for mutant in self.offsprings:
            if np.random.uniform() < self.mut_prob:
                if self.verbose is True:
                    Logger.log('  Performing mutation operation', 'INFO')
                self.toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness, i.e. those that have been mutated. The rest have not
        # changed ==> no need to re-evaluate them
        self.novel_offsprings = [ind for ind in self.offsprings if not ind.fitness.valid]
        if self.verbose is True:
            Logger.log(f'  {len(self.novel_offsprings)} novel offsprings found', 'INFO')

        # use only novel offsprings for evaluation
        # for the others we already know the outcome - no need to re-evaluate them
        self.offsprings_to_be_evaluated = deepcopy(self.novel_offsprings)
        self.latest_pop_size = len(self.offsprings_to_be_evaluated)

    def _ask(self):

        if self.num_generated == 0:
            raise NotImplementedError('the attribute "num_generated" was not expected to ever be zero here!')

        # If it is the first generation (note num_generated index starts from 1) we initialize the population
        if self.num_generated < 2:
            self._generate_first_population()
            self.offsprings = self.pop  # first iteration, so there is no offspring yet
            self.novel_offsprings = self.offsprings  # all offsprings are novel
            self.offsprings_to_be_evaluated = deepcopy(self.novel_offsprings)
        # otherwise select, mate and mutate population
        else:
            if len(self.offsprings_to_be_evaluated) == 0:
                # update population with previous offsprings and their associated observations/evaluations
                self._set_population_fitness()
                # cross-over and mutate, then select new offsprings to be evaluated
                # keep doing this if by chance we did not do any cross-over or mutation
                while len(self.offsprings_to_be_evaluated) == 0:
                    self._generate_offsprings()

        # evaluate one offspring at a time
        array = self.offsprings_to_be_evaluated.pop(0)
        return ParameterVector().from_array(array, self.param_space)

    @staticmethod
    def initIndividual(icls, bounds):
        """attribute generator: this defines how we initialize the population"""
        return icls(np.random.uniform(*p) for p in bounds)

    @staticmethod
    def project_bounds(bounds):
        """DEAP decorator to project out of bounds parameters back onto the boundary.
        `bounds` looks like this: [(low, high), (low, high), (low, high)]
        """
        def decorator(func):
            def wrapper(*args, **kargs):
                offspring = func(*args, **kargs)
                for child in offspring:
                    assert len(child) == len(bounds)
                    for i in range(len(child)):
                        if child[i] > bounds[i][1]:
                            child[i] = bounds[i][1]
                        elif child[i] < bounds[i][0]:
                            child[i] = bounds[i][0]
                return offspring
            return wrapper
        return decorator

