#!/usr/bin/env python

import numpy as np

from olympus.surfaces import AbstractSurface
from itertools import product

class CatAckley(AbstractSurface):

	""" The Ackley surface is inspired by the Ackley path function for
	continuous spaces. It features a narrow funnel around the global minimum,
	which is degenerate if the number of options along one (or more)dimensions
	is even and well-defined if the number of options for all dimensions is odd
	Ackley is to be evaluated on the hypercube
	x_i in [-32.768, 32.768] for i = 1, ..., d
	"""
	def __init__(self, param_dim, num_opts, descriptors=None):
		''' descriptors must be an iterable with the length num_opts
		For these surfaces, the same descriptors are used for each dimension
		'''
		value_dim = 1
		AbstractSurface.__init__(param_type='categorical', **locals())


	@property
	def minima(self):
		return None

	@property
	def maxima(self):
		return None

	def ackley(self, vector, a = 20., b = 0.2, c = 2. * np.pi):
		result = - a * np.exp( - b * np.sqrt( np.sum(vector**2) / self.param_dim) ) - np.exp( np.sum(np.cos(c * vector)) ) + a + np.exp(1)
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
			vector[index] = 65.536 * ( element / float(self.num_opts - 1) ) - 32.768
		return self.ackley(vector)
