#!/usr/bin/env python

import glob
import os
import sys

__home__ = os.path.dirname(os.path.abspath(__file__))


class ScalarizerLoader:
    def __init__(self, *args, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._find_scalarizers()

    def __getattr__(self, attr):
        if attr in ["Scalarizer", "AbstractScalarizer"]:
            attr_file = ScalarizerLoader.class_to_file(attr)
            module = __import__(
                f"olympus.scalarizers.{attr_file}", fromlist=[attr]
            )
            _class = getattr(module, attr)
            setattr(self, attr, _class)
            return _class
        else:
            scalarizer = ScalarizerLoader.import_scalarizer(attr)
            setattr(self, attr, scalarizer)
            return scalarizer

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
    def import_scalarizer(attr):
        attr_file = ScalarizerLoader.class_to_file(attr)
        module = __import__(
            f"olympus.scalarizers.scalarizer_{attr_file}", fromlist=[attr]
        )
        _class = getattr(module, attr)
        return _class

    def _find_scalarizers(self):
        self.scalarizer_files = []
        self.scalarizer_names = []
        self.scalarizer_map = {}
        for dir_name in glob.glob(f"{__home__}/scalarizer_*"):
            scalarizer_name = dir_name.split("/")[-1][11:]
            self.scalarizer_files.append(scalarizer_name)
            self.scalarizer_names.append(
                ScalarizerLoader.file_to_class(scalarizer_name)
            )

    def get_scalarizers_list(self):
        return sorted(self.scalarizer_names)


sys.modules[__name__] = ScalarizerLoader(**locals())
