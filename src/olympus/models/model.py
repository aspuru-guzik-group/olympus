#!/usr/bin/env python

from olympus import Logger
from . import get_models_list
from . import import_model
from . import AbstractModel


def Model(kind='Neuralnet'):
	"""Convenience function to access planners via a slightly higher level interface It returns a certain planner
	with defaults arguments by keyword.

	Args:
		kind (str or AbstractModel): keyword identifying one of the models available in Olympus. Alternatively,
			you can pass a custom model that is a subclass of AbstractModel.

	Returns:
		model: an instance of the chosen model.
	"""
	_validate_model_kind(kind)
	# if a string is passed, then load the corresponding wrapper
	if type(kind) == str:
		model = import_model(kind)
		model = model()
	# if a custom class is passed, then that is the 'wrapper'
	elif issubclass(kind, AbstractModel):
		model = kind()

	return model
	

def _validate_model_kind(kind):
	# if we received a string
	if type(kind) == str:
		avail_models = get_models_list()
		if kind not in avail_models:
			message = ('Model "{0}" not available in Olympus. Please choose '
					   'from one of the available models: {1}'.format(kind, ', '.join(avail_models)))
			Logger.log(message, 'FATAL')

	# if we received a custom model class
	elif issubclass(kind, AbstractModel):
		# make sure it has the necessary methods
		for method in ['_train', '_predict']:
			implementation = getattr(kind, method, None)
			if not callable(implementation):
				message = f'The object {kind} does not implement the necessary method "{method}"'
				Logger.log(message, 'FATAL')

	# if we do not know what was passed raise an error
	else:
		message = 'Could not initialize Model: the argument "kind" is neither a string or AbstractModel subclass'
		Logger.log(message, 'FATAL')
