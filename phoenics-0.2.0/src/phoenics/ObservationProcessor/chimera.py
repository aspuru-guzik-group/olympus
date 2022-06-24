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

The code in this file was developed at Harvard University (2018).
'''

__author__  = 'Florian Hase'

#=========================================================================

import numpy as np

#========================================================================

class Chimera(object):

	def __init__(self, tolerances, softness = 0.0, absolutes = None):
		self.tolerances = tolerances
		self.absolutes  = absolutes
		if absolutes is None:
			self.absolutes = np.zeros(len(tolerances)) + np.nan
		self.softness   = softness


	def soft_step(self, value):
		arg = - value / self.softness
		return 1. / (1. + np.exp(arg))


	def hard_step(self, value):
		result = np.empty(len(value))
		result = np.where(value > 0., 1., 0.)
		return result


	def step(self, value):
		if self.softness < 1e-5:
			return self.hard_step(value)
		else:
			return self.soft_step(value)


	def rescale(self, raw_objs):
		res_objs = np.empty(raw_objs.shape)
		res_abs  = np.empty(self.absolutes.shape)
		for index in range(raw_objs.shape[1]):
			min_objs, max_objs = np.amin(raw_objs[:, index]), np.amax(raw_objs[:, index])
			if min_objs < max_objs:
				res_abs[index]     = (self.absolutes[index] - min_objs) / (max_objs - min_objs)
				res_objs[:, index] = (raw_objs[:, index] - min_objs) / (max_objs - min_objs)
			else:
				res_abs[index]     = self.absolutes[index] - min_objs
				res_objs[:, index] = raw_objs[:, index] - min_objs
		return res_objs, res_abs				


	def shift_objectives(self, objs, res_abs):
		transposed_objs  = objs.transpose()
		shapes           = transposed_objs.shape
		shifted_objs     = np.empty((shapes[0] + 1, shapes[1]))
		
		mins, maxs, tols = [], [], []
		domain           = np.arange(shapes[1])
		shift            = 0
		for obj_index, obj in enumerate(transposed_objs):

			# get absolute tolerances
			minimum = np.amin(obj[domain])
			maximum = np.amax(obj[domain])
			mins.append(minimum)
			maxs.append(maximum)
			tolerance = minimum + self.tolerances[obj_index] * (maximum - minimum)
			if np.isnan(tolerance):
				tolerance = res_abs[obj_index]			

			# adjust region of interest
			interest = np.where(obj[domain] < tolerance)[0]
			if len(interest) > 0:
				domain = domain[interest]
	
			# apply shift	
			tols.append(tolerance + shift)
			shifted_objs[obj_index] = transposed_objs[obj_index] + shift

			# compute new shift
			if obj_index < len(transposed_objs) - 1:
				shift -= np.amax(transposed_objs[obj_index + 1][domain]) - tolerance
			else:
				shift -= np.amax(transposed_objs[0][domain]) - tolerance
				shifted_objs[obj_index + 1] = transposed_objs[0] + shift
		return shifted_objs, tols



	def scalarize_objs(self, shifted_objs, abs_tols):
		scalar_obj = shifted_objs[-1].copy()
		for index in range(0, len(shifted_objs) - 1)[::-1]:
			scalar_obj *= self.step( - shifted_objs[index] + abs_tols[index])
			scalar_obj += self.step(   shifted_objs[index] - abs_tols[index]) * shifted_objs[index]
		return scalar_obj.transpose()

	

	
	def scalarize(self, raw_objs):
		res_objs, res_abs      = self.rescale(raw_objs)
		shifted_objs, abs_tols = self.shift_objectives(res_objs, res_abs) 
		scalarized_obj         = self.scalarize_objs(shifted_objs, abs_tols)
		return scalarized_obj


#========================================================================

