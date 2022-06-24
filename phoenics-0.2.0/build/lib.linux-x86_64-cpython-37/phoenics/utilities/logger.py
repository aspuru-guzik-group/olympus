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
import traceback

#=========================================================================

class Logger(object):
	
	# DEBUG, INFO           --> stdout
	# WARNING, ERROR, FATAL --> stderr

	VERBOSITY_LEVELS = {-1: '',
						 0: ['INFO', 'FATAL'],
						 1: ['INFO', 'ERROR', 'FATAL'],
						 2: ['INFO', 'WARNING', 'ERROR', 'FATAL'],
						 3: ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'FATAL']}

	WRITER = {'DEBUG': sys.stdout, 'INFO': sys.stdout,
			  'WARNING': sys.stderr, 'ERROR': sys.stderr, 'FATAL': sys.stderr}

	# more colors and styles:
	# https://stackoverflow.com/questions/2048509/how-to-echo-with-different-colors-in-the-windows-command-line
	# https://joshtronic.com/2013/09/02/how-to-use-colors-in-command-line-output/

	GREY       = '0;37'
	WHITE      = '1;37'
	YELLOW     = '1;33',
	LIGHT_RED  = '1;31',
	RED        = '0;31'

	COLORS = {'DEBUG': WHITE, 'INFO': GREY, 'WARNING': YELLOW, 'ERROR': LIGHT_RED,'FATAL': RED}	

	def __init__(self, template, verbosity = 0):
		self.template         = template
		if isinstance(verbosity, dict):
			verbosity = verbosity['default']
		self.verbosity        = verbosity
		self.verbosity_levels = self.VERBOSITY_LEVELS[self.verbosity]


	def update_verbosity(self, verbosity = 0):
		if isinstance(verbosity, dict):
			verbosity = verbosity['default']
		self.verbosity        = verbosity
		self.verbosity_levels = self.VERBOSITY_LEVELS[self.verbosity]


	def log(self, message, message_type):
		
		# check if we need to log the message
		if message_type in self.verbosity_levels:
			color = self.COLORS[message_type]
			error_message = None
			if message_type in ['WARNING', 'ERROR', 'FATAL']:				
				error_message = traceback.format_exc()
				if not 'NoneType: None'in error_message:
					self.WRITER[message_type].write(error_message)
			uncolored_message = '[%s] %s ... %s ...\n' % (message_type, self.template, message)
			message           = "\x1b[%sm" % (color) + uncolored_message + "\x1b[0m"
			self.WRITER[message_type].write(message)
			return error_message, uncolored_message





