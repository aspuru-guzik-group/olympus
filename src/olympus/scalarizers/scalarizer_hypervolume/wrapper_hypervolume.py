#!/usr/bin/env python

import numpy as np

from olympus import Logger
from olympus.scalarizers import AbstractScalarizer
from olympus.utils.misc import get_hypervolume


class Hypervolume(AbstractScalarizer):
	""" Hypervolume indicator
	"""
	def __init__(self, value_space, goals):
		AbstractScalarizer.__init__(**locals())

		self.validate_asf_params()


	def scalarize(self, objectives):

		# flip signs for maximization objectives 
		signs = [1 if self.goals[idx]=='min' else -1 for idx in range(len(self.value_space))]
		objectives = objectives*signs
		w_ref = self.get_w_ref(objectives)

		merit = []
		for obs in objectives:
			obs = obs.reshape((1, obs.shape[0]))
			m_ = get_hypervolume(obs, w_ref)
			merit.append(m_)
		merit=np.array(merit) # larger merit is better

		if merit.shape[0] > 1:
			merit = self.normalize(merit.reshape(-1, 1))
			merit = np.squeeze(merit, axis=1)
		else:
			pass
		merit = 1.-merit  # smaller merit is better
		
		return merit

	@staticmethod
	def normalize(vector):
		min_ = np.amin(vector, axis=0)
		max_ = np.amax(vector, axis=0)
		ixs = np.where(np.abs(max_ - min_) < 1e-10)[0]
		if not ixs.size == 0:
			max_[ixs] = np.ones_like(ixs)
			min_[ixs] = np.zeros_like(ixs)
		return (vector - min_) / (max_ - min_)

	def get_w_ref(self, objectives):
		''' use the upper bound on all the objectives as the referenece point
		for the pareto hypervolume calculation
		'''
		return np.amax(objectives, axis=0)

	@staticmethod
	def check_kwargs(kwargs):
		"""quick and dirty check to see if the proper arguments are provided
		for the scalarizer
		"""
		req_args = ["goals"]
		provided_args = list(kwargs.keys())
		missing_args = list(set(req_args).difference(provided_args))
		if not missing_args == []:
			message = (
				f'Missing required Hypervolume arguments {", ".join(missing_args)}'
			)
			Logger.log(message, "FATAL")


	def validate_asf_params(self):
		pass
