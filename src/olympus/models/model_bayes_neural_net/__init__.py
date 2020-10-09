#!/usr/bin/env python

import traceback
from olympus import Logger

try:
    import silence_tensorflow

    silence_tensorflow.silence_tensorflow()
    import tensorflow
    import tensorflow_probability
except ModuleNotFoundError:
    # model_bnn throws ModuleNotFoundError exceptions for any module that
    # it depends on but that is not installed.
    # We catch the exception, determine the missing module and pass
    # it on to the user.
    error = traceback.format_exc()
    for line in error.split("\n"):
        if "ModuleNotFoundError" in line:
            module = line.strip().strip("'").split("'")[-1]

    message = """model_bnn requires {module}, which could not be found.
	Please install {module}.""".format(
        module=module
    )
    Logger.log(message, "FATAL")

# ========================================================================

from .wrapper_bayes_neural_net import BayesNeuralNet
