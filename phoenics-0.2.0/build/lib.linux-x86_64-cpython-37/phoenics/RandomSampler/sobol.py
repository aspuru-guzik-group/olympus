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

import sys
import numpy as np 

from utilities import PhoenicsModuleError

try:
	import sobol_seq
except ModuleNotFoundError:
	_, error_message, _ = sys.exc_info()
	extension = '\n\tTry installing the sobol_seq package or use uniform sampling instead.'
	PhoenicsModuleError(str(error_message) + extension)

#=========================================================================


class SobolContinuous(object):

	def __init__(self, seed = None):
		if seed is None:
			seed = np.random.randint(low = 0, high = 10**5)
		self.seed = seed


	def draw(self, low, high, size):
		num, dim = size[0], size[1]
		samples = []
		for _ in range(num):
			vector, seed = sobol_seq.i4_sobol(dim, self.seed)
			sample = (high - low) * vector + low
			self.seed = seed
			samples.append(sample)
		return np.array(samples)


#=========================================================================


