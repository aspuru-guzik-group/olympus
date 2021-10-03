#!/usr/bin/env python

import numpy as np


from olympus.surfaces import AbstractSurface
from itertools import product


class CatMichalewicz(AbstractSurface):

	"""
	The Michalewicz surface is generalized to categorical spaces from the Michalewicz function. This surface features well-defined options for each dimension which yield significantly better performances than
others. In addition, the number of pseudo-local minima scales factorially with the number of dimensions
	Michalewicz is to be evaluated on the hypercube
	x_i in [0, pi] for i = 1, ..., d
	"""
	def __init__(self, param_dim, num_opts, descriptors=None):
		'''
		'''
		AbstractSurface.__init__(param_type='categorical', **locals())


	@property
	def minima(self):
		return None

	@property
	def maxima(self):
		return None

	def michalewicz(self, vector, m = 10.):
		result = 0.
		for index, element in enumerate(vector):
			result += - np.sin(element) * np.sin( (index + 1) * element**2 / np.pi)**(2 * m)
		return result

	def _run(self, params):
		# map the sample onto the unit hypercube
		vector = np.zeros(self.param_dim)
		for index, element in enumerate(params):
			# make a messy check to see if the user passes integerts or
			# strings representing the categories
			# we expect either a int here, e.g. 12 or str of form e.g. 'x12'
			if isinstance(element, str):
				element = int(element[1:])
			elif isinstance(element, int):
				pass
			# TODO: add else statement here and return error
			vector[index] = np.pi * element / float(self.num_opts - 1)
		return self.michalewicz(vector)
