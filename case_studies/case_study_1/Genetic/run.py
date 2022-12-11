#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
import pickle
import os
import shutil
import itertools
from deap import base, creator, tools, algorithms

from olympus.datasets import Dataset

# --------
# Settings
# --------
budget = 200
repeats = 40


dataset = Dataset(kind='suzuki_edbo')


def run_experiment(ind):
    # convert indices to param names
    params = []
    for ix, p in enumerate(dataset.param_space):
        params.append(p.options[ind[ix]])
    return dataset.run(params, noiseless=True)


def eval_merit(ind):
    measurement = run_experiment(ind)
    return (-measurement[0][0],)



def save_pkl_file(data_all_repeats):
    """save pickle file with results so far"""

    if os.path.isfile('results.pkl'):
        shutil.move('results.pkl', 'bkp-results.pkl')  # overrides existing files

    # store run results to disk
    with open("results.pkl", "wb") as content:
        pickle.dump(data_all_repeats, content)


def load_data_from_pkl_and_continue(N):
    """load results from pickle file"""

    data_all_repeats = []
    # if no file, then we start from scratch/beginning
    if not os.path.isfile('results.pkl'):
        return data_all_repeats, N

    # else, we load previous results and continue
    with open("results.pkl", "rb") as content:
        data_all_repeats = pickle.load(content)

    missing_N = N - len(data_all_repeats)

    return data_all_repeats, missing_N

# --------------
# DEAP Functions
# --------------
def customMutation(individual, attrs_list, indpb=0.2, continuous_scale=0.1, discrete_scale=0.1):
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
                individual[i] = _project_bounds(individual[i], bound_low, bound_high)
            elif "discrete" in vartype:
                # add/substract an integer by rounding Gaussian perturbation
                # scale is 0.1 of domain range
                bound_low = attr.args[0]
                bound_high = attr.args[1]
                scale = (bound_high - bound_low) * discrete_scale
                delta = np.random.normal(loc=0.0, scale=scale)
                individual[i] += np.round(delta, decimals=0)
                individual[i] = _project_bounds(individual[i], bound_low, bound_high)
            elif "categorical" in vartype:
                # resample a random category
                individual[i] = attr()
            else:
                raise ValueError()
        else:
            continue

    return individual,


def apply_feasibility_constraint(child, parent, param_space):

        child_vector = np.array(child, dtype=object)
        feasible = True#known_constraints(child_vector)

        # if feasible, stop, no need to project the mutant
        if feasible is True:
            return

        # If not feasible, we try project parent or child onto feasibility boundary following these rules:
        # - for continuous parameters, we do stick breaking that is like a continuous version of a binary tree search
        #   until the norm of the vector connecting parent and child is less than a chosen threshold.
        # - for discrete parameters, we do the same until the "stick" is as short as possible, i.e. the next step
        #   makes it infeasible
        # - for categorical variables, we first reset them to the parent, then after having changed continuous
        #   and discrete, we reset the child. If feasible, we keep the child's categories, if still infeasible,
        #   we keep the parent's categories.

        parent_vector = np.array(parent, dtype=object)
        new_vector = child_vector

        continuous_mask = np.array([True if p['type'] == 'continuous' else False for p in param_space])
        categorical_mask = np.array([True if p['type'] == 'categorical' else False for p in param_space])

        child_continuous = child_vector[continuous_mask]
        child_categorical = child_vector[categorical_mask]

        parent_continuous = parent_vector[continuous_mask]
        parent_categorical = parent_vector[categorical_mask]

        # ---------------------------------------
        # (1) assign parent's categories to child
        # ---------------------------------------
        if any(categorical_mask) is True:
            new_vector[categorical_mask] = parent_categorical
            # If this fixes is, update child and return
            # This is equivalent to assigning the category to the child, and then going to step 2. Because child
            # and parent are both feasible, the procedure will converge to parent == child and will return parent
            if known_constraints(new_vector) is True:
                update_individual(child, new_vector)

        # -----------------------------------------------------------------------
        # (2) follow stick breaking/tree search procedure for continuous/discrete
        # -----------------------------------------------------------------------
        if any(continuous_mask) is True:
            # data needed to normalize continuous values
            lowers = np.array([d['low'] for d in param_space])
            uppers = np.array([d['high'] for d in param_space])
            inv_range = 1. / (uppers - lowers)
            counter = 0
            while True:
                # update continuous
                new_continuous = np.mean(np.array([parent_continuous, child_continuous]), axis=0)
                new_vector[continuous_mask] = new_continuous

                # if child is now feasible, parent becomes new_vector (we expect parent to always be feasible)
                if known_constraints(new_vector) is True:
                    parent_continuous = new_vector[continuous_mask]
                # if child still infeasible, child becomes new_vector (we expect parent to be the feasible one
                else:
                    child_continuous = new_vector[continuous_mask]

                # convergence criterion is that length of stick is less than 1% in all continuous dimensions
                parent_continuous_norm = (parent_continuous - lowers) * inv_range
                child_continuous_norm = (child_continuous - lowers) * inv_range
                # check all differences are within 1% of range
                if all(np.abs(parent_continuous_norm - child_continuous_norm) < 0.01):
                    break

                counter += 1
                if counter > 150:  # convergence above should be reached in 128 iterations max
                    raise ValueError("constrained evolution procedure ran into trouble")

        # last parent values are the feasible ones
        new_vector[continuous_mask] = parent_continuous

        # ---------------------------------------------------------
        # (3) Try reset child's categories, otherwise keep parent's
        # ---------------------------------------------------------
        if any(categorical_mask) is True:
            new_vector[categorical_mask] = child_categorical
            if known_constraints(new_vector) is True:
                update_individual(child, new_vector)
                return
            else:
                # This HAS to be feasible, otherwise there is a bug
                new_vector[categorical_mask] = parent_categorical
                update_individual(child, new_vector)
                return
        else:
            update_individual(child, new_vector)
            return


def update_individual(ind, value_vector):
    for i, v in enumerate(value_vector):
        ind[i] = v


def cxDummy(ind1, ind2):
    """Dummy crossover that does nothing. This is used when we have a single gene in the chromosomes, such that
    crossover would not change the population.
    """
    return ind1, ind2


def create_deap_toolbox(param_space):
    from deap import base

    toolbox = base.Toolbox()
    attrs_list = []

    for i, param in enumerate(param_space):
        vartype = param['type']

        if vartype in 'continuous':
            toolbox.register(f"x{i}_{vartype}", np.random.uniform, param['low'], param['high'])

        elif vartype in 'discrete':
            toolbox.register(f"x{i}_{vartype}", np.random.randint, param['low'], param['high'])

        elif vartype in 'categorical':
            toolbox.register(f"x{i}_{vartype}", np.random.choice, param['categories'])

        attr = getattr(toolbox, f"x{i}_{vartype}")
        attrs_list.append(attr)

    return toolbox, attrs_list


def _project_bounds(x, x_low, x_high):
    if x < x_low:
        return x_low
    elif x > x_high:
        return x_high
    else:
        return x


def param_vectors_to_deap_population(param_vectors):
    population = []
    for param_vector in param_vectors:
        ind = creator.Individual(param_vector)
        population.append(ind)
    return population


def update_offspring_fitnesses(pop, X_samples, chimera_fitness):
    assert len(chimera_fitness) == len(X_samples)
    for ind in pop:
        idx = X_samples.index(ind)
        # assign the corresponding fitness
        fit = chimera_fitness[idx]
        ind.fitness.values = fit,


def build_set_of_feasible_samples(n=10):
    # frags_idxs = df_results.loc[:, ['r1_label', 'r3_label', 'r4_label', 'r5_label']].to_numpy()
    # np.random.shuffle(frags_idxs)
    # return frags_idxs

    param_options = [list(np.arange(len(p.options))) for p in dataset.param_space]
    cart_product = np.array(list(itertools.product(*param_options)))
    np.random.shuffle(cart_product)
    print(cart_product.shape)
    return cart_product[:10, :]




electrophile_options = np.arange(len(dataset.param_space[0].options))
nucleophile_options = np.arange(len(dataset.param_space[1].options))
base_options = np.arange(len(dataset.param_space[2].options))
ligand_options = np.arange(len(dataset.param_space[3].options))
solvent_options = np.arange(len(dataset.param_space[4].options))


param_space = [
               {'type': 'categorical', 'categories': electrophile_options},
               {'type': 'categorical', 'categories': nucleophile_options},
               {'type': 'categorical', 'categories': base_options},
               {'type': 'categorical', 'categories': ligand_options},
               {'type': 'categorical', 'categories': solvent_options},
            ]

# setup GA with DEAP
creator.create("FitnessMax", base.Fitness, weights=[-1.0])  # we minimize
creator.create("Individual", list, fitness=creator.FitnessMax)

# make toolbox
toolbox, attrs_list = create_deap_toolbox(param_space)

# crossover and mutation probabilites
CXPB, MUTPB = 0.5, 0.5

# check whether we are appending to previous results
data_all_repeats, missing_repeats = load_data_from_pkl_and_continue(repeats)



for num_repeat in range(missing_repeats):

    print(f'   Repeat {len(data_all_repeats)+1}')

    # initialize with 10 random samples
    samples = build_set_of_feasible_samples(n=10)
    X_samples = []
    y_samples = []

    toolbox.register("population", param_vectors_to_deap_population)
    #toolbox.register("evaluate", eval_merit)
    # use custom mutations for continuous, discrete, and categorical variables
    toolbox.register("mutate", customMutation, attrs_list=attrs_list, indpb=0.5)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # mating type depends on how many genes we have
    if np.shape(samples)[1] == 1:
        toolbox.register("mate", cxDummy)  # i.e. no crossover
    elif np.shape(samples)[1] == 2:
        toolbox.register("mate", tools.cxUniform, indpb=0.5)  # uniform crossover
    else:
        toolbox.register("mate", tools.cxTwoPoint)  # two-point crossover

    # Initialise population
    ngen = 0
    population = toolbox.population(samples)

    # add individuals to X_samples and y_samples
    for ind in population:
        X_samples.append(ind)
        y_samples.append(run_experiment(ind))

    # Evaluate pop fitnesses
    fitnesses = np.array(y_samples)#chimera.scalarize(np.array(y_samples))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit,

    # create hall of fame
    num_elites = 2  # 2 elite individuals in each population (i.e. always keep best)
    halloffame = tools.HallOfFame(num_elites)  # hall of fame with top individuals
    halloffame.update(population)

    # fits = [ind.fitness.values[0] for ind in population]
    # print('fits :', )
    # print('ngen : ', ngen)
    # print('population :', population )
    # print('X_samples : ', X_samples)
    # print header info
    # print("{0:>10}{1:>10}{2:>10}{3:>10}{4:>10}{5:>10}{6:>10}".format('gen', 'neval', 'nsamples', 'avg', 'std', 'min', 'max'))
    # print("{0:>10d}{1:>10}{2:>10}{3:>10.4f}{4:>10.4f}{5:>10.4f}{6:>10.4f}".format(ngen, len(population), len(X_samples),
    #                                                                           np.mean(fits), np.std(fits),
    #                                                                           min(fits), max(fits)))

    # keep evaluating new offprings until we exhaust budget
    while len(X_samples) < budget:
        ngen += 1

        # size of hall of fame
        hof_size = len(halloffame.items) if halloffame.items else 0

        # Select the next generation individuals (allow for elitism)
        offspring = toolbox.select(population, len(population) - hof_size)

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < CXPB:
                parent1 = list(map(toolbox.clone, child1))  # both are parents to both children, but we select one here
                parent2 = list(map(toolbox.clone, child2))
                # mate
                toolbox.mate(child1, child2)
                # apply constraints
                #apply_feasibility_constraint(child1, parent1, param_space)
                #apply_feasibility_constraint(child2, parent2, param_space)
                # clear fitness values
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.random() < MUTPB:
                parent = list(map(toolbox.clone, mutant))
                # mutate
                toolbox.mutate(mutant)
                # apply constraints
                #apply_feasibility_constraint(mutant, parent, param_space)
                # clear fitness values
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        neval = len(invalid_ind) # how many new offprings we are evaluating

        # append new offsprings to X_samples and y_samples
        for ind in invalid_ind:
            # append only new samples
            if ind not in X_samples:
                X_samples.append(ind)
                y_samples.append(run_experiment(ind))
                if len(X_samples) >= budget:
                    break

        # re-evaluate offsprings objectives
        offspring_objs = [run_experiment(ind) for ind in offspring]

        # get chimera fitnesses for new offsprings
        offspring_fitnesses = np.array(offspring_objs) #chimera.scalarize(np.array(offspring_objs))

        # re-assign fitness for all offsprings (not just the new ones, as we need to rank the individuals at next iteration)
        for ind, fit in zip(offspring, offspring_fitnesses):
            ind.fitness.values = fit,

        # add the best back to population
        offspring.extend(halloffame.items)

        # Update the hall of fame with the generated individuals
        halloffame.update(offspring)

        # replace the old population with the new offsprings
        population[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        # fits = [ind.fitness.values for ind in population]
        # print("{0:>10d}{1:>10}{2:>10}{3:>10.4f}{4:>10.4f}{5:>10.4f}{6:>10.4f}".format(ngen, neval, len(X_samples),
        #                                                                           np.mean(fits), np.std(fits),
        #                                                                           min(fits), max(fits)))

        # if we are over the budget, stop
        if len(X_samples) >= budget:
            break

    # store run results into a DataFrame
    electrophile = [x[0] for x in X_samples]
    nucleophile = [x[1] for x in X_samples]
    base = [x[2] for x in X_samples]
    ligand = [x[3] for x in X_samples]
    solvent = [x[4] for x in X_samples]

    yield_ = [y[0][0] for y in y_samples]
    data = pd.DataFrame({ 'electrophile': electrophile, 'nucleophile': nucleophile, 'base': base, 'ligand': ligand,  'yield': yield_})
    data_all_repeats.append(data)

    # save results to disk
    save_pkl_file(data_all_repeats)
