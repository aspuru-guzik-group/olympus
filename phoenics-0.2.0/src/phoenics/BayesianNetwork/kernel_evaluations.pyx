#!/usr/bin/env python 

'''
Licensed to the Apache Software Foundation (ASF) under one or more 
contributor license agreements. See the NOTICE file distributed with this 
work for additional information regarding copyright ownership. The ASF 
licenses this file to you under the Apache License, Version 2.0 (the 
"License"); you may not use this file except in compliance with the 
License. You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT 
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the 
License for the specific language governing permissions and limitations 
under the License.

The code in this file was developed at Harvard University (2018) and 
modified at ChemOS Inc. (2019) as stated in the NOTICE file.
'''

__author__  = 'Florian Hase'

#=========================================================================

import  cython 
cimport cython

from cython.parallel import prange

import  numpy as np 
cimport numpy as np 

from libc.math cimport exp, abs, round

#=========================================================================

cdef class KernelEvaluator:

	cdef int    num_samples, num_obs, num_kernels
	cdef double lower_prob_bound, inv_vol
	
	cdef np.ndarray np_locs, np_sqrt_precs
	cdef np.ndarray np_objs
	cdef np.ndarray np_probs

	def __init__(self, locs, sqrt_precs, lower_prob_bound, objs, inv_vol):
		
		self.np_locs          = locs
		self.np_sqrt_precs    = sqrt_precs
		self.np_objs          = objs

		self.num_samples      = locs.shape[0]
		self.num_obs          = locs.shape[1]
		self.num_kernels      = locs.shape[2]
		self.lower_prob_bound = lower_prob_bound
		self.inv_vol          = inv_vol

		self.np_probs = np.zeros(self.num_obs, dtype = np.float64)


	cdef double [:] _probs(self, double [:] sample):

		cdef int    sample_index, obs_index, kernel_index
		cdef int    num_indices
		cdef double total_prob, prec_prod, exp_arg_sum

		cdef double [:, :, :] locs       = self.np_locs
		cdef double [:, :, :] sqrt_precs = self.np_sqrt_precs 
		cdef double inv_sqrt_two_pi = 0.3989422804014327

		cdef double [:] probs = self.np_probs
		for obs_index in range(self.num_obs):
			probs[obs_index] = 0.

		cdef double obs_probs

		for obs_index in range(self.num_obs):
			obs_probs = 0.

			for sample_index in range(self.num_samples):
				total_prob   = 1.
				prec_prod    = 1.
				exp_arg_sum  = 0.
				kernel_index = 0

				while kernel_index < self.num_kernels:

					prec_prod     = prec_prod * sqrt_precs[sample_index, obs_index, kernel_index]
					exp_arg_sum   = exp_arg_sum + (sqrt_precs[sample_index, obs_index, kernel_index] * (sample[kernel_index] - locs[sample_index, obs_index, kernel_index]))**2
					kernel_index += 1

				obs_probs += total_prob * prec_prod * exp( - 0.5 * exp_arg_sum)

				if sample_index == 100:
					if 0.01 * obs_probs * inv_sqrt_two_pi**self.num_kernels < self.lower_prob_bound:
						probs[obs_index] = 0.01 * obs_probs * inv_sqrt_two_pi**self.num_kernels / self.num_samples
						break
			else:
				probs[obs_index] = obs_probs * inv_sqrt_two_pi**self.num_kernels / self.num_samples
		return probs


#	@cython.boundscheck(False)
	cpdef get_kernel(self, np.ndarray sample):

		cdef int obs_index
		cdef double temp_0, temp_1
		cdef double inv_den
		
		cdef double [:] sample_memview = sample
		probs_sample = self._probs(sample_memview)

		# construct numerator and denominator of acquisition
		cdef double num = 0.
		cdef double den = 0.
		cdef double [:] objs = self.np_objs 

		for obs_index in range(self.num_obs):
			temp_0 = objs[obs_index]
			temp_1 = probs_sample[obs_index]
			num += temp_0 * temp_1
			den += temp_1

		inv_den = 1. / (self.inv_vol + den)

		return num, inv_den, probs_sample
	





