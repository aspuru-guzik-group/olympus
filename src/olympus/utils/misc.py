#!/usr/bin/env python

#===============================================================================

import uuid
import traceback

#===============================================================================

def generate_id():
    identifier = str(uuid.uuid4())[:8]
    return identifier

#===============================================================================

def check_module(module_name, message, **kwargs):
    try:
        _ = __import__(module_name)
    except ModuleNotFoundError:
        from olympus import Logger
        error = traceback.format_exc()
        for line in error.split('\n'):
            if 'ModuleNotFoundError' in line:
                module = line.strip().strip("'").split("'")[-1]
        kwargs.update(locals())
        message = f'{message}'.format(**kwargs)
        Logger.log(message, 'ERROR')

def check_planner_module(planner, module, link):
    from olympus.planners import get_planners_list
    message = '''Planner {planner} requires {module} which could not be found. Please install
    {module} and check out {link} for further instructions or use another
    available planner: {planners_list}'''
    check_module(module, message, planner=planner, planners_list=', '.join(get_planners_list()), link=link)

#===============================================================================
