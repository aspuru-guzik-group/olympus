#!/usr/bin/env python

# ===============================================================================

import json
import os
from abc import ABCMeta as NativeABCMeta

from olympus import Logger


# ====================
# Olympus Object class
# ====================
class Object:

    """Abstraction of a dictionary

    - facilitates declaration of defaults
    - can be extended to enable further modifications
    """

    def __init__(self, me="", indent=0, **kwargs):
        """creates empty object and loads defaults

        Args:
            me     (str): arbitrary name to identify the object
            indent (int): number of spaces used in string
                representation
        """
        self.me = me
        self.indent = indent

        self.props = []
        self.attrs = []
        self.max_prop_len = 0

        # get defaults
        for prop in dir(self):
            if prop.startswith("ATT_"):
                attr = getattr(self, prop).copy()
                default = attr["default"]
                if callable(default):
                    default = attr["default"]()
                self.add(prop[4:].lower(), default)

        # overwrite defaults with any kwargs we have received
        for key, value in kwargs.items():
            if "ATT_{}".format(key.upper()) in dir(self) or not key in self:
                self.add(key, value)
            else:
                message = f"Attribute {key} is already defined for {self}"
                Logger.log(message, "WARNING")

        valid = self._validate()
        if not valid:
            message = "Could not validate {}".format(str(self))
            Logger.log(message, "WARNING")

    def __iter__(self):
        for _, prop in enumerate(self.props):
            yield prop, self.attrs[_]

    def __contains__(self, prop):
        return prop in self.props

    def __getattr__(self, prop):
        prop_name = prop[4:]
        if prop.startswith("set_"):

            def set_attr(attr):
                setattr(self, prop_name, attr)

            return set_attr
        elif prop.startswith("get_"):
            self.add(prop, lambda: self.get(prop_name))
            return getattr(self, prop)
        elif prop == "__getstate__":
            return lambda: self.__dict__
        elif prop == "__setstate__":

            def set_state(_dict):
                self.__dict__ = _dict

            return set_state
        elif f"ATT_{prop.upper()}" in dir(self):
            return getattr(self, f"ATT_{prop.upper()}")["default"]
        raise AttributeError(f"Object has no attribute {prop}")

    def __getitem__(self, prop):
        return getattr(self, prop)

    def __setitem__(self, prop, attr):
        self.add(prop, attr)

    def __str__(self):
        string = "{}\n".format(self.me)
        indent = " " * self.indent
        for prop in sorted(self.props):
            string += "{ind}--> {prop: <{width}}{attr}\n".format(
                ind=indent,
                prop=prop + ":",
                width=self.max_prop_len + 2,
                attr=getattr(self, prop),
            )
        return string[:-1]

    @property
    def defaults(self):
        defaults = []
        for prop in dir(self):
            if prop.startswith("ATT_"):
                attr_dict = getattr(self, prop)
                attr_dict["name"] = prop[4:]
                defaults.append(attr_dict)
        return defaults

    def _validate(self, **kwargs):
        for key, value in kwargs.items():
            if (
                not value
                in getattr(self, "ATT_{}".format(key.upper()))["valid"]
            ):
                return False
        return True

    def add(self, prop, attr):
        """dynamically adds property and attribute to object

        Args:
            prop (any): property associated with attribute
            attr (any): property value
        """

        if prop in ["props", "attrs"]:
            return

        setattr(self, prop, attr)
        if prop in self.props:
            self.attrs[self.props.index(prop)] = attr
        else:
            self.props.append(prop)
            self.attrs.append(attr)
            self.max_prop_len = max(len(prop), self.max_prop_len)

    def get(self, prop):
        """returns attribute associated with given property

        Args:
            prop (any): valid property

        Returns:
            any: attribute associated with property
        """
        return getattr(self, prop)

    def update(self, prop, attr):
        self.add(prop, attr)

    def from_dict(self, info_dict):
        """returns object representation of given dictionary

        Args:
            info_dict (dict): dictionary to be represented

        Returns:
            Object: Object representation of dictionary
        """
        for key, value in info_dict.items():
            self.add(key, value)
        return self

    def to_dict(self, exclude=[]):
        """returns dictionary representation of presented object

        Args:
            exclude (list of any): properties to be excluded

        Returns:
            dict: representation of presented object
        """
        info_dict = {}
        for prop in sorted(self.props):
            if prop in exclude:
                continue
            # ATTENTION: perhaps check for callables instead
            if prop.startswith("get_"):
                continue
            if prop.startswith("set_"):
                continue

            attr = getattr(self, prop)
            if isinstance(attr, Object):
                attr = attr.to_dict()
            if isinstance(attr, list):
                converted = []
                for elem in attr:
                    if isinstance(elem, Object):
                        elem = elem.to_dict()
                    converted.append(elem)
                attr = converted
            info_dict[prop] = attr
        return info_dict

    def from_json(self, json_file="config.json"):
        if os.path.isfile(json_file):
            with open(json_file, "r") as content:
                config = json.loads(content.read())
        else:
            config = json.loads(json_file)
        self.from_dict(info_dict=config)
        return self

    def to_json(self, json_file="config.json"):
        config_dict = self.to_dict()
        #        with open(json_file, 'w') as fp:
        config_json = json.dumps(config_dict, sort_keys=True, indent=4)
        return config_json

    def reset(self):
        self.props = []
        self.attrs = []


# ============================
# Abstract Attribute Decorator
# ============================
def abstract_attribute(obj=None):
    if obj is None:
        obj = DummyAttribute()
    obj.__is_abstract_attribute__ = True
    return obj


class DummyAttribute:
    pass


class ABCMeta(NativeABCMeta):
    def __call__(cls, *args, **kwargs):
        instance = NativeABCMeta.__call__(cls, *args, **kwargs)
        abstract_attributes = {
            name
            for name in dir(instance)
            if getattr(
                getattr(instance, name), "__is_abstract_attribute__", False
            )
        }
        if abstract_attributes:
            raise NotImplementedError(
                "Can't instantiate {0} without abstract "
                "attributes: {1}".format(
                    cls.__name__, ", ".join(abstract_attributes)
                )
            )
        return instance
