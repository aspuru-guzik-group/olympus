#!/usr/bin/env python

#===============================================================================

import uuid
import traceback

#===============================================================================

def generate_id():
    identifier = str(uuid.uuid4())[:8]
    return identifier

#===============================================================================

def check_module(module_name, message, **kwargs):
    try:
        _ = __import__(module_name)
    except ModuleNotFoundError:
        from olympus import Logger
        error = traceback.format_exc()
        for line in error.split('\n'):
            if 'ModuleNotFoundError' in line:
                module = line.strip().strip("'").split("'")[-1]
        kwargs.update(locals())
        message = f'{message}'.format(**kwargs)
        Logger.log(message, 'ERROR')

def check_planner_module(planner, module, link):
    from olympus.planners import get_planners_list
    message = '''Planner {planner} requires {module} which could not be found. Please install
    {module} and check out {link} for further instructions or use another
    available planner: {planners_list}'''
    check_module(module, message, planner=planner, planners_list=', '.join(get_planners_list()), link=link)

#===============================================================================

def r2_score(y_true, y_pred):
    """:math:`R^2` (coefficient of determination) regression score function.
    Best possible score is 1.0 and it can be negative (because the
    model can be arbitrarily worse). A constant model that always
    predicts the expected value of y, disregarding the input features,
    would get a :math:`R^2` score of 0.0.

    This function is taken from sklearn, and is used when sklearn is not available.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    Returns
    -------
    z : float or ndarray of floats
        The :math:`R^2` score or ndarray of scores if 'multioutput' is
        'raw_values'.

    Notes
    -----
    This is not a symmetric function.
    Unlike most other scores, :math:`R^2` score may be negative (it need not
    actually be the square of a quantity R).
    This metric is not well-defined for single samples and will return a NaN
    value if n_samples is less than two.
    References
    ----------
    .. [1] `Wikipedia entry on the Coefficient of determination
            <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_
    """
    import numpy as np

    if len(y_pred) < 2:
        # TODO use proper logging
        print("R^2 score is not well-defined with less than two samples.")
        return np.nan

    if len(y_true.shape) == 1:
        y_true = y_true.reshape((len(y_true), 1)) # unsqueeze
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape((len(y_true), 1)) # unsqueeze

    numerator = ((y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
    denominator = ((y_true - np.average( y_true, axis=0)) ** 2).sum(axis=0, dtype=np.float64)
    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    output_scores = np.ones([y_true.shape[1]])
    output_scores[valid_score] = 1 - (numerator[valid_score] /
                                      denominator[valid_score])
    # arbitrary set to zero to avoid -inf scores, having a constant
    # y_true is not interesting for scoring a regression anyway
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.
    return np.average(output_scores)