#!/usr/bin/env python

import os, sys
import time

planners = ['RandomSearch', 'Hyperopt', 'Gpyopt', 'Gryffin', 'Dragonfly', 'Botorch', 'Hebo']

datasets = ['suzuki', 'suzuki_edbo', 'suzuki_i', 'suzuki_ii', 'suzuki_iii', 'suzuki_iv']

cwd = os.getcwd()

for planner in planners:
	for dataset in datasets:

		os.chdir(f'{planner}/{dataset}')
		os.system('pwd')
		# submit job
		#os.system('sbatch submit.sh')
		#time.sleep(2)

		os.chdir(cwd)