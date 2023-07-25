#!/usr/bin/env python

import glob
import os
import sys

__home__ = os.path.dirname(os.path.abspath(__file__))

# ===============================================================================


class PlannerLoader:
    def __init__(self, *args, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._find_planners()

    def __getattr__(self, attr):
        if attr in ["Planner", "AbstractPlanner", "CustomPlanner"]:
            attr_file = PlannerLoader.class_to_file(attr)
            module = __import__(
                f"olympus.planners.{attr_file}", fromlist=[attr]
            )
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
                file_name += f"_{character.lower()}"
            else:
                file_name += character
        return file_name

    @staticmethod
    def file_to_class(file_name):
        class_name = file_name[0].upper()
        next_upper = False
        for character in file_name[1:]:
            if character == "_":
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
        module = __import__(
            f"olympus.planners.planner_{attr_file}", fromlist=[attr]
        )
        _class = getattr(module, attr)
        return _class

    @staticmethod
    def get_param_types(name):
        file = PlannerLoader.class_to_file(name)
        module = __import__(
            f"olympus.planners.planner_{file}", fromlist=[name]
        )
        return module.param_types

    def _find_planners(self):
        self.planner_files = []
        self.planner_names = []
        self.planner_param_types = []
        self.planners_map = {}
        for dir_name in glob.glob(f"{__home__}/planner_*"):

            if "/" in dir_name:
                planner_name = dir_name.split("/")[-1][8:]
            elif "\\" in dir_name:
                planner_name = dir_name.split("/")[-1][8:]

            self.planner_files.append(planner_name)
            self.planner_names.append(
                PlannerLoader.file_to_class(planner_name)
            )

    def get_planners_list(self):
        return sorted(self.planner_names)

    def get_cont_planners_list(self):
        """return list of all planner which can handle continuous variables"""
        planner_param_types = [
            PlannerLoader.get_param_types(name) for name in self.planner_names
        ]
        return [
            planner
            for planner, types in zip(self.planner_names, planner_param_types)
            if "continuous" in types
        ]

    def get_disc_planners_list(self):
        """return list of all planners which can handle discrete variables"""
        planner_param_types = [
            PlannerLoader.get_param_types(name) for name in self.planner_names
        ]
        return [
            planner
            for planner, types in zip(self.planner_names, planner_param_types)
            if "discrete" in types
        ]

    def get_cat_planners_list(self):
        """return list of all planners which can handle categorical variables"""
        planner_param_types = [
            PlannerLoader.get_param_types(name) for name in self.planner_names
        ]
        return [
            planner
            for planner, types in zip(self.planner_names, planner_param_types)
            if "categorical" in types
        ]

    def list_planners(self):
        return sorted(self.planner_names)


# ===============================================================================

sys.modules[__name__] = PlannerLoader(**locals())

# ===============================================================================
