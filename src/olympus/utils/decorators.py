#!/usr/bin/env python

import functools
import threading

#===============================================================================

def thread(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target = function, args = args, kwargs = kwargs)
        thread.start()
    return wrapper

def daemon(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target = function, args = args, kwargs = kwargs)
        thread.daemon = True
        thread.start()
    return wrapper
    
#===============================================================================
