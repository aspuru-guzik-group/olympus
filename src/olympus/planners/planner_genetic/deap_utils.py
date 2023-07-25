#!/usr/bin/env python

from deap import algorithms, base, creator, tools

data_all_repeats, missing_repeats = load_data_from_pkl_and_continue(repeats)

# --------------
# DEAP Functions
# --------------
def customMutation(
    individual, attrs_list, indpb=0.2, continuous_scale=0.1, discrete_scale=0.1
):
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
                individual[i] = _project_bounds(
                    individual[i], bound_low, bound_high
                )
            elif "discrete" in vartype:
                # add/substract an integer by rounding Gaussian perturbation
                # scale is 0.1 of domain range
                bound_low = attr.args[0]
                bound_high = attr.args[1]
                scale = (bound_high - bound_low) * discrete_scale
                delta = np.random.normal(loc=0.0, scale=scale)
                individual[i] += np.round(delta, decimals=0)
                individual[i] = _project_bounds(
                    individual[i], bound_low, bound_high
                )
            elif "categorical" in vartype:
                # resample a random category
                individual[i] = attr()
            else:
                raise ValueError()
        else:
            continue

    return (individual,)


def apply_feasibility_constraint(child, parent, param_space):

    child_vector = np.array(child)
    child_dict = {"x0": child_vector[0], "x1": child_vector[1]}
    feasible = surface.eval_constr(child_dict)

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

    parent_vector = np.array(parent)
    new_vector = child_vector

    continuous_mask = np.array(
        [True if p["type"] == "continuous" else False for p in param_space]
    )
    categorical_mask = np.array(
        [True if p["type"] == "categorical" else False for p in param_space]
    )

    child_continuous = child_vector[continuous_mask]
    child_categorical = child_vector[categorical_mask]

    parent_continuous = parent_vector[continuous_mask]
    parent_categorical = parent_vector[categorical_mask]

    # ---------------------------------------
    # (1) assign parent's categories to child
    # ---------------------------------------
    if any(categorical_mask) is True:
        new_vector[categorical_mask] = parent_categorical
        new_dict = {"x0": new_vector[0], "x1": new_vector[1]}
        # If this fixes is, update child and return
        # This is equivalent to assigning the category to the child, and then going to step 2. Because child
        # and parent are both feasible, the procedure will converge to parent == child and will return parent
        if surface.eval_constr(new_dict) is True:
            update_individual(child, new_vector)
            return

    # -----------------------------------------------------------------------
    # (2) follow stick breaking/tree search procedure for continuous/discrete
    # -----------------------------------------------------------------------
    if any(continuous_mask) is True:
        # data needed to normalize continuous values
        lowers = np.array([d["low"] for d in param_space])
        uppers = np.array([d["high"] for d in param_space])
        inv_range = 1.0 / (uppers - lowers)
        counter = 0
        while True:
            # update continuous
            new_continuous = np.mean(
                np.array([parent_continuous, child_continuous]), axis=0
            )
            new_vector[continuous_mask] = new_continuous
            new_dict = {"x0": new_vector[0], "x1": new_vector[1]}

            # if child is now feasible, parent becomes new_vector (we expect parent to always be feasible)
            if surface.eval_constr(new_dict) is True:
                parent_continuous = new_vector[continuous_mask]
            # if child still infeasible, child becomes new_vector (we expect parent to be the feasible one
            else:
                child_continuous = new_vector[continuous_mask]

            # convergence criterion is that length of stick is less than 1% in all continuous dimensions
            parent_continuous_norm = (parent_continuous - lowers) * inv_range
            child_continuous_norm = (child_continuous - lowers) * inv_range
            # check all differences are within 1% of range
            if all(
                np.abs(parent_continuous_norm - child_continuous_norm) < 0.01
            ):
                break

            counter += 1
            if (
                counter > 150
            ):  # convergence above should be reached in 128 iterations max
                raise ValueError(
                    "constrained evolution procedure ran into trouble"
                )

    # last parent values are the feasible ones
    new_vector[continuous_mask] = parent_continuous

    # ---------------------------------------------------------
    # (3) Try reset child's categories, otherwise keep parent's
    # ---------------------------------------------------------
    if any(categorical_mask) is True:
        new_vector[categorical_mask] = child_categorical
        new_dict = {"x0": new_vector[0], "x1": new_vector[1]}
        if surface.eval_constr(new_dict) is True:
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
        vartype = param["type"]

        if vartype in "continuous":
            toolbox.register(
                f"x{i}_{vartype}",
                np.random.uniform,
                param["low"],
                param["high"],
            )

        elif vartype in "discrete":
            toolbox.register(
                f"x{i}_{vartype}",
                np.random.randint,
                param["low"],
                param["high"],
            )

        elif vartype in "categorical":
            toolbox.register(
                f"x{i}_{vartype}", np.random.choice, param["categories"]
            )

        attr = getattr(toolbox, f"x{i}_{vartype}")
        attrs_list.append(attr)

    return toolbox, attrs_list


def project_bounds(x, x_low, x_high):
    if x < x_low:
        return x_low
    elif x > x_high:
        return x_high
    else:
        return x


# def param_vectors_to_deap_population(param_vectors):
#     population = []
#     for param_vector in param_vectors:
#         ind = creator.Individual(param_vector)
#         population.append(ind)
#     return population


def propose_randomly(num_proposals, param_space):
    """Randomly generate num_proposals proposals. Returns the numerical
    representation of the proposals as well as the string based representation
    for the categorical variables

    Args:
            num_proposals (int): the number of random proposals to generate
    """
    proposals = []
    raw_proposals = []
    for propsal_ix in range(num_proposals):
        sample = []
        raw_sample = []
        for param_ix, param in enumerate(param_space):
            if param.type == "continuous":
                p = np.random.uniform(param.low, param.high, size=None)
                sample.append(p)
                raw_sample.append(p)
            elif param.type == "discrete":
                num_options = int(
                    ((param.high - param.low) / param.stride) + 1
                )
                options = np.linspace(param.low, param.high, num_options)
                p = np.random.choice(options, size=None, replace=False)
                sample.append(p)
                raw_sample.append(p)
            elif param.type == "categorical":
                options = param.options
                p = np.random.choice(options, size=None, replace=False)
                feat = cat_param_to_feat(param, p)
                sample.extend(feat)  # extend because feat is vector
                raw_sample.append(p)
        proposals.append(sample)
        raw_proposals.append(raw_sample)
    proposals = np.array(proposals)
    raw_proposals = np.array(raw_proposals)

    return proposals, raw_proposals
