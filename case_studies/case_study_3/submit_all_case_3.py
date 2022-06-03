#!/usr/bin/env python

import os, sys
import time

planners = ['RandomSearch', 'Gpyopt', 'Gryffin', 'Botorch'] # 'Dragonfly'

scalarizers = ['Parego', 'Chimera', 'WeightedSum', 'Hypervolume']

datasets = ['redoxmers', 'dye_lasers']

cwd = os.getcwd()

for planner in planners:
	for dataset in datasets:
		for scalarizer in scalarizers:

			os.chdir(f'{planner}/{dataset}/{scalarizer}')
			os.system('pwd')
			# submit job
			os.system('pwd')
			#os.system('sbatch submit.sh')
			#time.sleep(2)

			os.chdir(cwd)
