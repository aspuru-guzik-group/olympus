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
  
__author__ = 'Florian Hase'

#========================================================================

import numpy as np 

#========================================================================

class AbstractOptimizer(object):

	dx = 1e-6

	def __init__(self, func, *args, **kwargs):
		self.func = func
		for key, value in kwargs.items():
			setattr(self, str(key), value)


	def _set_func(self, func, pos = None):
		self.func = func
		if pos is not None:
			self.pos     = pos
			self.num_pos = len(pos)


	def grad(self, sample, step = None):
		if step is None: step = self.dx
		gradients = np.zeros(len(sample), dtype = np.float32)
		perturb   = np.zeros(len(sample), dtype = np.float32)
		for pos_index, pos in enumerate(self.pos):
			if pos is None: continue
			perturb[pos] += step
			gradient = (self.func(sample + perturb) - self.func(sample - perturb)) / (2. * step)
			gradients[pos] = gradient
			perturb[pos] -= step
		return gradients
	

