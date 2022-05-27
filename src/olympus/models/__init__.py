#!/usr/bin/env python

import glob
import os
import sys

__home__ = os.path.dirname(os.path.abspath(__file__))


class ModelLoader:
    def __init__(self, *args, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._find_models()

    def __getattr__(self, attr):
        if attr in ["Model", "AbstractModel"] or attr.startswith("Wrapper"):
            attr_file = ModelLoader.class_to_file(attr)
            module = __import__(f"olympus.models.{attr_file}", fromlist=[attr])
            _class = getattr(module, attr)
            setattr(self, attr, _class)
            return _class
        else:
            model = ModelLoader.import_model(attr)
            setattr(self, attr, model)
            return model

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
    def import_model(attr):
        attr_file = ModelLoader.class_to_file(attr)
        module = __import__(
            f"olympus.models.model_{attr_file}", fromlist=[attr]
        )
        _class = getattr(module, attr)
        return _class

    def _find_models(self):
        self.model_files = []
        self.model_names = []
        self.model_map = {}
        for dir_name in glob.glob(f"{__home__}/model_*"):
            model_name = dir_name.split("/")[-1][6:]
            self.model_files.append(model_name)
            self.model_names.append(ModelLoader.file_to_class(model_name))

    def get_models_list(self):
        return sorted(self.model_names)


sys.modules[__name__] = ModelLoader(**locals())
