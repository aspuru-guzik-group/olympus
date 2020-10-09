#!/usr/bin/env python

#======================================================================

import traceback

from olympus import Logger

#======================================================================

try:
	import smac
except ModuleNotFoundError:

	# GPyOpt throws ModuleNotFoundError exceptions for any module that
	# it depends on but that is not installed.
	# We catch the exception, determine the missing module and pass
	# it on to the user.
	error = traceback.format_exc()
	for line in error.split('\n'):
		if 'ModuleNotFoundError' in line:
			module = line.strip().strip("'").split("'")[-1]

	message = '''SMAC requires {module}, which could not be found.
	Please install {module} and check out
	https://www.automl.org/automated-algorithm-design/algorithm-configuration/smac/ for further instructions'''.format(module = module)
	Logger.log(message, 'FATAL')

#======================================================================

from .wrapper_smac import Smac
