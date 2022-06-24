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

#=========================================================================

import numpy as np 

try:
	from Acquisition.NumpyOptimizers import AbstractOptimizer
except ModuleNotFoundError:
	from abstract_optimizer import AbstractOptimizer

#=========================================================================

class AdamOptimizer(AbstractOptimizer):

	iterations = 0

	def __init__(self, 
			func = None, 
			eta = 0.1, 
			beta_1 = 0.9, 
			beta_2 = 0.999, 
			epsilon = 1e-8, 
			decay = 0., 
			*args, **kwargs):
	
		AbstractOptimizer.__init__(self, func, *args, **kwargs)
		self.eta     = eta
		self.beta_1  = beta_1
		self.beta_2  = beta_2
		self.epsilon = epsilon
		self.decay   = decay
		self.initial_decay = self.decay
		self.ms, self.vs = None, None


	def set_func(self, func, pos = None):
		self.iterations = 0
		self._set_func(func, pos)
		if hasattr(self, 'ms'):
			delattr(self, 'ms')
			delattr(self, 'vs')

	def get_update(self, vector):
		grads = self.grad(vector)
		eta   = self.eta
		if self.initial_decay > 0.:
			eta *= (1. / (1. + self.decay * self.iterations))
		
		next_iter = self.iterations + 1
		eta_next  = eta * (np.sqrt(1. - np.power(self.beta_2, next_iter)) / (1. - np.power(self.beta_1, next_iter)))
		
		if not hasattr(self, 'ms'):
			self.ms = [0. for element in vector]
			self.vs = [0. for element in vector]

		update = np.zeros(len(vector)).astype(np.float32) + np.nan
		for index, element, grad, mass, vel in zip(range(len(vector)), vector, grads, self.ms, self.vs):
			m_next = (self.beta_1 * mass) + (1. - self.beta_1) * grad
			v_next = (self.beta_2 * vel)  + (1. - self.beta_2) * np.square(grad)
			p_next = element - eta_next * m_next / (np.sqrt(v_next) + self.epsilon)
			self.ms[index] = m_next
			self.vs[index] = v_next
			update[index]  = p_next

		self.iterations += 1
		return np.array(update)

#=========================================================================

if __name__ == '__main__':

	import matplotlib.pyplot as plt 
	import seaborn as sns 

	adam = AdamOptimizer()

	def func(x):
		return (x - 1)**2

	adam.set_func(func, pos = np.arange(1))

	domain = np.linspace(-1, 3, 200)
	values = func(domain)

	start = np.zeros(1) - 0.8

	plt.ion()

	for _ in range(10**3):
		
		plt.clf()
		plt.plot(domain, values)
		plt.plot(start, func(start), marker = 'o', color = 'k')

		start = adam.get_update(start)

		plt.pause(0.05)

#		print(start, func(start))




