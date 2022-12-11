#!/usr/bin/env python


import numpy as np

from olympus import Logger
from olympus.scalarizers import AbstractScalarizer


class R2(AbstractScalarizer):
	"""R2 indicator acheivement scalarizing function"""

	def __init__(self, value_space, goals, num_weights=50):
		AbstractScalarizer.__init__(**locals())

		self.validate_asf_params()

	def scalarize(self, objectives):

		signs = [1 if self.goals[idx]=='min' else -1 for idx in range(len(self.value_space))]
		objectives = objectives*signs
		# norm_objectives = self.normalize(objectives)
		w_ref = np.amin(objectives, axis=0)

		W = np.random.sample((self.num_weights, len(self.value_space)))
		sum_W = np.sum(W, axis=1).reshape(-1, 1)
		W = W / sum_W

		merit = []
		for objective in objectives:
			objective = objective.reshape((1, objective.shape[0]))
			sum_terms = []
			for w in W:
				t = w*np.abs(objective-w_ref)
				t_max = np.amax(t, axis=0)
				min_t_max = np.amin(t_max, axis=0)
				sum_terms.append(min_t_max)
			m = (1./self.num_weights)*np.sum(sum_terms)
			merit.append(m)
		merit = np.array(merit)

		if merit.shape[0] > 1:
			# normalize the merit (best value is 0., worst is 1.)
			merit = self.normalize(merit.reshape(-1, 1))
			return np.squeeze(merit, axis=1)
		else:
			return merit

	@staticmethod
	def normalize(vector):
		min_ = np.amin(vector)
		max_ = np.amax(vector)
		ixs = np.where(np.abs(max_ - min_) < 1e-10)[0]
		if not ixs.size == 0:
			max_[ixs] = np.ones_like(ixs)
			min_[ixs] = np.zeros_like(ixs)
		return (vector - min_) / (max_ - min_)

	def validate_asf_params(self):
		pass

	@staticmethod
	def check_kwargs(kwargs):
		"""quick and dirty check to see if the proper arguments are provided
		for the scalarizer
		"""
		req_args = []
		provided_args = list(kwargs.keys())
		missing_args = list(set(req_args).difference(provided_args))
		if not missing_args == []:
			message = (
				f'Missing required R2 arguments {", ".join(missing_args)}'
			)
			Logger.log(message, "FATAL")


if __name__ == "__main__":

	from olympus import Surface

	surf = Surface(kind="MultFonseca")

	scalarizer = R2(surf.value_space, goals=["max", "max"])

	objectives = np.array([[0.1, 0.4], [0.7, 0.9], [0.04, 0.08]])

	merit = scalarizer.scalarize(objectives)

	print('merit : ', merit)
