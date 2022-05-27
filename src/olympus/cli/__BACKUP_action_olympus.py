#!/usr/bin/env python

import abc
import argparse

# ===============================================================================


class ActionOlympus(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        argparse.Action.__init__(self, option_strings, dest, **kwargs)

    def __call__(self, parser, name_space, values, option_string=None):
        setattr(name_space, self.dest, values)
        print("VALUES", values)
        self._action(values)

    @abc.abstractmethod
    def _action(self, *args):
        pass


# ===============================================================================
