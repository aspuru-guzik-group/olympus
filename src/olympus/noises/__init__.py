#!/usr/bin/env python

import glob
import os
import sys

__home__ = os.path.dirname(os.path.abspath(__file__))


class NoiseLoader:
    def __init__(self, *args, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._find_noise()

    def __getattr__(self, attr):
        if attr in ["Noise", "AbstractNoise"]:
            attr_file = NoiseLoader.class_to_file(attr)
            module = __import__(f"olympus.noises.{attr_file}", fromlist=[attr])
            _class = getattr(module, attr)
            setattr(self, attr, _class)
            return _class
        else:
            noise = NoiseLoader.import_noise(attr)
            setattr(self, attr, noise)
            return noise

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
    def import_noise(attr):
        attr_file = NoiseLoader.class_to_file(attr)
        module = __import__(
            f"olympus.noises.noise_{attr_file}", fromlist=[attr]
        )
        _class = getattr(module, attr)
        return _class

    def _find_noise(self):
        self.noise_files = []
        self.noise_names = []
        self.noise_map = {}
        for dir_name in glob.glob(f"{__home__}/noise_*"):
            noise_name = dir_name.split("/")[-1][6:]
            self.noise_files.append(noise_name)
            self.noise_names.append(NoiseLoader.file_to_class(noise_name))

    def get_noises_list(self):
        return sorted(self.noise_names)

    def list_noises(self):
        return sorted(self.noise_names)


sys.modules[__name__] = NoiseLoader(**locals())
