#!/usr/bin/env python

import glob
import os
import sys

__home__ = os.path.dirname(os.path.abspath(__file__))


class SurfaceLoader:
    def __init__(self, *args, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._find_surfaces()

    def __getattr__(self, attr):
        if attr in ["Surface", "AbstractSurface"]:
            attr_file = SurfaceLoader.class_to_file(attr)
            module = __import__(
                f"olympus.surfaces.{attr_file}", fromlist=[attr]
            )
            _class = getattr(module, attr)
            setattr(self, attr, _class)
            return _class
        else:
            surface = SurfaceLoader.import_surface(attr)
            setattr(self, attr, surface)
            return surface

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
    def import_surface(attr):
        attr_file = SurfaceLoader.class_to_file(attr)
        module = __import__(
            f"olympus.surfaces.surface_{attr_file}", fromlist=[attr]
        )
        _class = getattr(module, attr)
        return _class

    def _find_surfaces(self):
        self.surface_files = []
        self.surface_names = []
        self.surfaces_map = {}
        for dir_name in glob.glob(f"{__home__}/surface_*"):
            surface_name = dir_name.split("/")[-1][8:]
            self.surface_files.append(surface_name)
            self.surface_names.append(
                SurfaceLoader.file_to_class(surface_name)
            )

    def get_surfaces_list(self):
        return sorted(self.surface_names)

    def list_surfaces(self):
        return sorted(self.surface_names)

    def list_cont_surfaces(self):
        all_surfaces = self.list_surfaces()
        return [surf for surf in all_surfaces if not surf.startswith("Cat")]

    def list_cat_surfaces(self):
        all_surfaces = self.list_surfaces()
        return [surf for surf in all_surfaces if surf.startswith("Cat")]

    def list_moo_surfaces(self):
        all_surfaces = self.list_surfaces()
        return [surf for surf in all_surfaces if surf.startswith("Mult")]

    def list_so_surfaces(self):
        all_surfaces = self.list_surfaces()
        return [surf for surf in all_surfaces if not surf.startswith("Mult")]


sys.modules[__name__] = SurfaceLoader(**locals())
