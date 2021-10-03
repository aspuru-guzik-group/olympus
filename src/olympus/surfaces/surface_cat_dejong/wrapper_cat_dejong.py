#!/usr/bin/env python

import numpy as np


from olympus.surfaces import AbstractSurface
from itertools import product


class CatDejong(AbstractSurface):

		""" Categorical version of the Dejong surface
		To be evaluated on the hypercube
		x_i in [-5.12, 5.12] for i = 1, ..., d
		"""
		def __init__(self, param_dim, num_opts, descriptors=None):
			''' descriptors must be an iterable with the length num_opts
			For these surfaces, the same descriptors are used for each dimension
			'''
			AbstractSurface.__init__(param_type='categorical', **locals())

		@property
		def minima(self):
			return None

		@property
		def maxima(self):
			return None

		def dejong(sef, vector):
			result = np.sum(vector**2)
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
				vector[index] = 10.24 * ( element / float(self.num_opts - 1) ) - 5.12
			return self.dejong(vector)
