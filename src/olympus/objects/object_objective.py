#!/usr/bin/env python 

#======================================================================

from olympus.objects import Object

#======================================================================

class ObjectObjective(Object):

	ATT_NAME = {'type': 'name',   'default': 'objective'}
	ATT_GOAL = {'type': 'string', 'default': 'minimize', 'valid': ['minimize', 'maximize']}

#======================================================================
