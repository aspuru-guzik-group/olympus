#!/usr/bin/env python

from copy import deepcopy

import numpy as np
from deap import base, creator, tools

from olympus import Logger
from olympus.objects import ParameterVector
from olympus.planners import AbstractPlanner


# ==========
# Main Class
# ==========
class Genetic(AbstractPlanner):

	PARAM_TYPES = ["continuous"]

	def __init__(
		self,
		goal="minimize",
		pop_size=10,
		cx_prob=0.5,
		mut_prob=0.2,
		random_seed=None,
		verbose=False,
		mate_args={"function": tools.cxTwoPoint},
		mutate_args={
			"function": tools.mutGaussian,
			"mu": 0,
			"sigma": 0.2,
			"indpb": 0.2,
		},
		select_args={"function": tools.selTournament, "tournsize": 3},
	):
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
		creator.create(
			"FitnessMin", base.Fitness, weights=(-1.0,)
		)  # -1 for minimisation, +1 for maximisation
		creator.create("Individual", list, fitness=creator.FitnessMin)




	def create_deap_toolbox(self, param_space):
		from deap import base

		toolbox = base.Toolbox()
		attrs_list = []

		for i, param in enumerate(param_space):
			vartype = param['type']

			if vartype in 'continuous':
				toolbox.register(f"x{i}_{vartype}", np.random.uniform, param['domain'][0], param['domain'][1])

			elif vartype in 'discrete':
				toolbox.register(f"x{i}_{vartype}", np.random.randint, param['domain'][0], param['domain'][1])

			elif vartype in 'categorical':
				toolbox.register(f"x{i}_{vartype}", np.random.choice, param['categories'])

			attr = getattr(toolbox, f"x{i}_{vartype}")
			attrs_list.append(attr)

		return toolbox, attrs_list


	def reset(self):
		"""Clears the current population so that a new optimization would start from a randomly initialised population
		again.
		"""
		if hasattr(self, "pop"):
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
		if len(self._param_space) == 1:
			message = 'Resorting to cxDummy mating with parameter dimension of 1'
			Logger.log(message, 'WARNING')
			self.toolbox.register("mate", self.cxDummy)
		elif len(self._param_space) == 2:
			message = 'Resorting to cxUniform mating with parameter dimension of 2'
			Logger.log(message, 'WARNING')
			self.toolbox.register('mate', tools.cxUniform, indpb=0.5)
		else:
			self.toolbox.register("mate", **self.select_args)

		# ----------------
		# mutate operation
		# ----------------
		# custom mulations for continuous, discrete and categorical variables
		# TODO change indpb to a user argument
		self.toolbox.register("mutate", self.customMutation, attrs_list=self.attrs_list, indpb=0.5)

		# ----------------
		# select operation
		# ----------------
		self.toolbox.register("select", **self.select_args)

		# ----------------------------------
		# decorate mate and mutate operations
		# -----------------------------------
		# bounds = [param["domain"] for param in self._param_space]
		# self.toolbox.decorate("mate", self.project_bounds(bounds))
		# self.toolbox.decorate("mutate", self.project_bounds(bounds))

		# Alternative option to control the bounds:
		# self.toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
		# self.toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0,
		#                       indpb=1.0/len(self._param_space))



	def _set_param_space(self, param_space):
		self._param_space = []
		for param in param_space:
			if param.type == "continuous":
				param_dict = {
					"name": param.name,
					"type": param.type,
					"domain": (param.low, param.high),
				}
			elif param.type == 'categorical':
				param_dict = {
					"name": param.name,
					"type": param.type,
					"categories": param.options,
				}
			self._param_space.append(param_dict)

		# get Deap toolbox and attr list ()
		self.toolbox, self.attrs_list = self.create_deap_toolbox(self._param_space)

		# register here as they depend on the param_space
		self._register_deap_operations()



	def _tell(self, observations):
		self._params = observations.get_params(as_array=False)
		self._values = observations.get_values(
			as_array=True, opposite=self.flip_measurements
		)
		# reshape the array so to have shape (num observations, num objectives)
		# WARNING: In this case we are sticking to 1 objective, multiple objective are not supported!
		# TODO: fix the following reshaping to allow multiple objectives
		self._values = np.reshape(
			self._values, newshape=(np.shape(self._values)[0], 1)
		)


	def propose_randomly(self, num_proposals):
		"""Randomly generate num_proposals proposals. Returns the numerical
		representation of the proposals as well as the string based representation
		for the categorical variables

		Args:
				num_proposals (int): the number of random proposals to generate
		"""
		proposals = []
		for propsal_ix in range(num_proposals):
			sample = []
			for param_ix, param in enumerate(self.param_space):
				if param.type == "continuous":
					p = np.random.uniform(param.low, param.high, size=None)
					sample.append(p)
				elif param.type == "discrete":
					num_options = int(
						((param.low - param.high) / 1) + 1
					)
					options = np.linspace(param.low, param.high, num_options)
					p = np.random.choice(options, size=None, replace=False)
					sample.append(p)
				elif param.type == "categorical":
					options = param.options
					p = np.random.choice(options, size=None, replace=False)
					sample.append(p)
			proposals.append(sample)

		return proposals

	def _generate_first_population(self):
		if self.verbose is True:
			Logger.log(
				f"Creating first population of size {self.pop_size}", "INFO"
			)

		# Structure initializers
		# bounds = [param["domain"] for param in self._param_space]
		# self.toolbox.register(
		#     "individual",
		#     self.initIndividual,
		#     icls=creator.Individual,
		#     bounds=bounds,
		# )

		# self.toolbox.register(
		#     "population", tools.initRepeat, list, self.toolbox.individual
		# )
		self.toolbox.register("population", self.param_vectors_to_deap_population)
		#self.pop = self.toolbox.population(n=self.pop_size)
		
		# generate the initial samples
		samples = self.propose_randomly(num_proposals=self.pop_size)
		self.pop = self.toolbox.population(samples)

		self.latest_pop_size = self.pop_size

		# delete creator classes
		del creator.Individual
		del creator.FitnessMin

	def _set_population_fitness(self):

		# assign evaluations to each offspring
		offsprings_fitness = self._values[-self.latest_pop_size :, :]
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
			Logger.log("Creating new offsprings...", "INFO")

		# Select the next generation individuals
		self.offsprings = self.toolbox.select(self.pop, len(self.pop))
		# Clone the selected individuals
		self.offsprings = list(map(self.toolbox.clone, self.offsprings))

		# Apply crossover and on the offsprings
		for child1, child2 in zip(self.offsprings[::2], self.offsprings[1::2]):
			if np.random.uniform() < self.cx_prob:
				if self.verbose is True:
					Logger.log("  Performing cross-over operation", "INFO")
				self.toolbox.mate(child1, child2) 
				del child1.fitness.values
				del child2.fitness.values

		# Apply mutation on the offsprings
		for mutant in self.offsprings:
			if np.random.uniform() < self.mut_prob:
				if self.verbose is True:
					Logger.log("  Performing mutation operation", "INFO")
				self.toolbox.mutate(mutant)
				del mutant.fitness.values

		# Evaluate the individuals with an invalid fitness, i.e. those that have been mutated. The rest have not
		# changed ==> no need to re-evaluate them
		self.novel_offsprings = [
			ind for ind in self.offsprings if not ind.fitness.valid
		]
		if self.verbose is True:
			Logger.log(
				f"  {len(self.novel_offsprings)} novel offsprings found",
				"INFO",
			)

		# use only novel offsprings for evaluation
		# for the others we already know the outcome - no need to re-evaluate them
		self.offsprings_to_be_evaluated = deepcopy(self.novel_offsprings)
		self.latest_pop_size = len(self.offsprings_to_be_evaluated)

	def _ask(self):

		if self.num_generated == 0:
			raise NotImplementedError(
				'the attribute "num_generated" was not expected to ever be zero here!'
			)

		# If it is the first generation (note num_generated index starts from 1) we initialize the population
		if self.num_generated < 2:
			self._generate_first_population()
			self.offsprings = (
				self.pop
			)  # first iteration, so there is no offspring yet
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

	# @staticmethod
	# def project_bounds(bounds):
	#     """DEAP decorator to project out of bounds parameters back onto the boundary.
	#     `bounds` looks like this: [(low, high), (low, high), (low, high)]
	#     """

	#     def decorator(func):
	#         def wrapper(*args, **kargs):
	#             offspring = func(*args, **kargs)
	#             for child in offspring:
	#                 assert len(child) == len(bounds)
	#                 for i in range(len(child)):
	#                     if child[i] > bounds[i][1]:
	#                         child[i] = bounds[i][1]
	#                     elif child[i] < bounds[i][0]:
	#                         child[i] = bounds[i][0]
	#             return offspring

	#         return wrapper

	#     return decorator

	@staticmethod
	def _project_bounds(x, x_low, x_high):
		if x < x_low:
			return x_low
		elif x > x_high:
			return x_high
		else:
			return x

	def customMutation(self, individual, attrs_list, indpb=0.2, continuous_scale=0.1, discrete_scale=0.1):
		"""Mutation
		Parameters
		----------
		indpb : float
			Independent probability for each attribute to be mutated.
		"""
		assert len(individual) == len(attrs_list)

		for i, attr in enumerate(attrs_list):
			# determine whether we are performing a mutation
			if np.random.random() < indpb:
				vartype = attr.__name__
				if "continuous" in vartype:
					# Gaussian perturbation with scale being 0.1 of domain range
					bound_low = attr.args[0]
					bound_high = attr.args[1]
					scale = (bound_high - bound_low) * continuous_scale
					individual[i] += np.random.normal(loc=0.0, scale=scale)
					individual[i] = self._project_bounds(individual[i], bound_low, bound_high)
				elif "discrete" in vartype:
					# add/substract an integer by rounding Gaussian perturbation
					# scale is 0.1 of domain range
					bound_low = attr.args[0]
					bound_high = attr.args[1]
					scale = (bound_high - bound_low) * discrete_scale
					delta = np.random.normal(loc=0.0, scale=scale)
					individual[i] += np.round(delta, decimals=0)
					individual[i] = self._project_bounds(individual[i], bound_low, bound_high)
				elif "categorical" in vartype:
					# resample a random category
					individual[i] = attr()
				else:
					raise ValueError()
			else:
				continue

		return individual


	def param_vectors_to_deap_population(self, param_vectors):
		population = []
		for param_vector in param_vectors:
			ind = creator.Individual(param_vector)
			population.append(ind)
		return population

	@staticmethod
	def cxDummy(ind1, ind2):
		"""Dummy crossover that does nothing. This is used when we have a single gene in the chromosomes, such that
		crossover would not change the population.
		"""
		return ind1, ind2


# -----------
# DEBUGGING
# -----------
if __name__ == '__main__':
	PARAM_TYPE = "categorical"

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

		planner = Genetic(goal="minimize")
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

		planner = Genetic(goal="minimize")
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

		planner = Genetic(goal="minimize")
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
