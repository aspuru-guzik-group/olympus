#!/usr/bin/env python

# ===============================================================================

import traceback
import uuid

import itertools
import numpy as np
import numpy.matlib



from ._r2_score import r2_score


# ===============================================================================


def generate_id():
    identifier = str(uuid.uuid4())[:8]
    return identifier


# ===============================================================================


def check_module(module_name, message, **kwargs):
    try:
        _ = __import__(module_name)
    except ModuleNotFoundError:
        from olympus import Logger

        error = traceback.format_exc()
        for line in error.split("\n"):
            if "ModuleNotFoundError" in line:
                module = line.strip().strip("'").split("'")[-1]
        kwargs.update(locals())
        message = f"{message}".format(**kwargs)
        Logger.log(message, "ERROR")


def check_planner_module(planner, module, link):
    from olympus.planners import get_planners_list

    message = """Planner {planner} requires {module} which could not be found. Please install
    {module} and check out {link} for further instructions or use another
    available planner: {planners_list}"""
    check_module(
        module,
        message,
        planner=planner,
        planners_list=", ".join(get_planners_list()),
        link=link,
    )


def get_pareto(objs):
    ''' Determine the pareto set of parameters and the pareto front
    of objectives

    Implementation inspired by: https://github.com/shinya-ml/Multiobj-Bayes-opt

    Args:
        params (np.ndarray): input parameters 
        objs (np.ndarray): objectives
    '''
    objs_copy = np.copy(objs)
    value_idx = 0
    pareto_front = np.empty((0, objs.shape[1]))

    while value_idx < objs_copy.shape[0]:
        objs_out = np.delete(objs_copy, value_idx, axis=0)
        flag = np.all(objs_out <= objs_copy[value_idx, :], axis=1)
        if not np.any(flag):
            pareto_front = np.append(pareto_front, [objs_copy[value_idx, :]], axis=0)
            value_idx += 1
        else:
            objs_copy = np.delete(objs_copy, value_idx, axis=0)
    
    return pareto_front

def get_pareto_set(params, objs):
    ''' Determine the pareto set of parameters and the pareto front
    of objectives

    Implementation inspired by: https://github.com/shinya-ml/Multiobj-Bayes-opt

    Args:
        params (np.ndarray): input parameters 
        objs (np.ndarray): objectives
    '''
    objs_copy = np.copy(objs)
    value_idx = 0
    param_idx = 0
    pareto_front = np.empty((0, objs.shape[1]))
    pareto_set   = np.empty((0, params.shape[1]))

    while value_idx < objs_copy.shape[0]:
        objs_out = np.delete(objs_copy, value_idx, axis=0)
        flag = np.all(objs_out <= objs_copy[value_idx, :], axis=1)
        if not np.any(flag):
            pareto_front = np.append(pareto_front, [objs_copy[value_idx, :]], axis=0)
            pareto_set = np.append(pareto_set, [params[param_idx, :]], axis=0)
            value_idx += 1
        else:
            objs_copy = np.delete(objs_copy, value_idx, axis=0)
        param_idx += 1
   
    return pareto_front, pareto_set


def get_hypervolume(objs, w_ref):
    ''' compute hypervolume indicator. Importantly, this assumes that
    all objectives have a goal of 'minimize'. The user must flip the signs 
    of 'maximize' objectives before calling this function.

    Implementation inspired by: https://github.com/shinya-ml/Multiobj-Bayes-opt
    
    Args:
        objs (np.ndarray): objective values
        w_ref (np.ndarray): reference point for hypervolume indicator
    '''
    hypervolume = 0.0
    pareto_front = get_pareto(objs)
    v, w = get_cells(pareto_front, w_ref)

    if v.ndim == 1:
        hypervolume = np.prod(w - v)
    else:
        hypervolume = np.sum(np.prod(w - v, axis=1))
    return hypervolume



def get_cells(pareto_front, ref, ref_inv=None):
    ''' create cells for hypervolume indicator calculation

    Implementation inspired by: https://github.com/shinya-ml/Multiobj-Bayes-opt
    
    Args:
        pareto_front (np.ndarray): pareto front objective points
        ref (np.array): reference point for pareto hypervolume 
            calculation 

    '''
    n, l = np.shape(pareto_front)

    if ref_inv is None:
        ref_inv = np.min(pareto_front, axis=0)

    if n == 1: 
        return np.atleast_2d(pareto_front), np.atleast_2d(ref)
    else:
        hv = np.prod(pareto_front - ref, axis=1)
        pivot_idx = np.argmax(hv)
        pivot = pareto_front[pivot_idx]

        lower = np.atleast_2d(pivot)
        upper = np.atleast_2d(ref)

        for i in itertools.product(range(2), repeat=l):
            iter_idx = np.array(list(i))==0
            if  (np.sum(iter_idx) == 0) or (np.sum(iter_idx) == l):
                continue

            new_ref = pivot.copy()
            new_ref[iter_idx] = ref[iter_idx]

            new_ref_inv = ref_inv.copy()
            new_ref_inv[iter_idx] = pivot[iter_idx]

            new_pareto_front = pareto_front[(pareto_front < new_ref).all(axis=1), :]
            new_pareto_front[new_pareto_front < new_ref_inv] = np.matlib.repmat(
                new_ref_inv, new_pareto_front.shape[0], 1
                )[new_pareto_front < new_ref_inv]

            if np.size(new_pareto_front) > 0:
                child_lower, child_upper = get_cells(new_pareto_front, new_ref, new_ref_inv)

                lower = np.r_[lower, np.atleast_2d(child_lower)]
                upper = np.r_[upper, np.atleast_2d(child_upper)]

    return lower, upper