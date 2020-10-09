#!/usr/bin/env python

import os, sys, glob
__home__ = os.path.dirname(os.path.abspath(__file__))

#===============================================================================

class PlannerLoader:

    def __init__(self, *args, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._find_planners()

    def __getattr__(self, attr):
        if attr in ['Planner', 'AbstractPlanner', 'CustomPlanner']:
            attr_file = PlannerLoader.class_to_file(attr)
            module = __import__(f'olympus.planners.{attr_file}', fromlist=[attr])
            _class = getattr(module, attr)
            setattr(self, attr, _class)
            return _class
        else:
            planner = PlannerLoader.import_planner(attr)
            setattr(self, attr, planner)
            return planner

    @staticmethod
    def class_to_file(class_name):
        file_name = class_name[0].lower()
        for character in class_name[1:]:
            if character.isupper():
                file_name += f'_{character.lower()}'
            else:
                file_name += character
        return file_name


    @staticmethod
    def file_to_class(file_name):
        class_name = file_name[0].upper()
        next_upper = False
        for character in file_name[1:]:
            if character == '_':
                next_upper = True
                continue
            if next_upper:
                class_name += character.upper()
            else:
                class_name += character
            next_upper = False
        return class_name


    @staticmethod
    def import_planner(attr):
        attr_file = PlannerLoader.class_to_file(attr)
        module = __import__(f'olympus.planners.planner_{attr_file}', fromlist=[attr])
        _class = getattr(module, attr)
        return _class


    def _find_planners(self):
        self.planner_files = []
        self.planner_names = []
        self.planners_map  = {}
        for dir_name in glob.glob(f'{__home__}/planner_*'):
            planner_name = dir_name.split('/')[-1][8:]
            self.planner_files.append(planner_name)
            self.planner_names.append(PlannerLoader.file_to_class(planner_name))


    def get_planners_list(self):
        return sorted(self.planner_names)

    def list_planners(self):
        return sorted(self.planner_names)

#===============================================================================

sys.modules[__name__] = PlannerLoader(**locals())

#===============================================================================
