#!/usr/bin/env python

import numpy as np


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

    return proposals, raw_proposals


def cat_param_to_feat(param, val):
    """convert the option selection of a categorical variable (usually encoded
    as a string) to a machine readable feature vector

    Args:
            param (object): the categorical olympus parameter
            val (str): the value of the chosen categorical option
    """
    # get the index of the selected value amongst the options
    arg_val = param.options.index(val)
    if np.all([d == None for d in param.descriptors]):
        # no provided descriptors, resort to one-hot encoding
        # feat = np.array([arg_val])
        feat = np.zeros(len(param.options))
        feat[arg_val] += 1.0
    else:
        # we have descriptors, use them as the features
        feat = param.descriptors[arg_val]
    return feat
