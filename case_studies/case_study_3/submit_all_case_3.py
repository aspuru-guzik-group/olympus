#!/usr/bin/env python

import os, sys
import time

planners = ['RandomSearch', 'Hyperopt', 'Gpyopt', 'Gryffin', 'Dragonfly', 'Botorch', 'Hebo']

scalarizers = ['Chimera', 'WeightedSum', 'Parego']

cwd = os.getcwd()

for planner in planners:
	for scalarizer in scalarizers:

		os.chdir(f'{planner}/{scalarizer}')
		os.system('pwd')
		# submit job
		os.system('sbatch submit.sh')
		time.sleep(2)

		os.chdir(cwd)