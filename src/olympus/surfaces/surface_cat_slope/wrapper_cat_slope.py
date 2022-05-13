#!/usr/bin/env python

import numpy as np


from olympus.surfaces import AbstractSurface
from itertools import product


class CatSlope(AbstractSurface):

	""" The Slope surface is constructed such that the response linearly increases with the index of the
option along each dimension in the reference ordering. As such, the Slope surface presents a generalization of a plane
to categorical domains
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

	def slope(self, vector):
		seed   = 0
		vector = np.array(vector)
		for index, element in enumerate(vector):
			# check to see if strings are passed
			if isinstance(element, str):
				element = int(element[1:])
			elif isinstance(element, int):
				pass
			seed += self.num_opts**index * element
		print(vector)
		result = np.sum(vector / self.num_opts)
		return result

	def _run(self, params):
		return self.slope(params)
