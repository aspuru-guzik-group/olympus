#!/usr/bin/env python


import numpy as np
from olympus import Logger
from olympus.objects import Object


class ObjectParameter(Object):

    ATT_NAME = {"type": "string", "default": "parameter"}




class ObjectParameterContinuous(ObjectParameter):

    ATT_TYPE = {
        "type": "string",
        "default": "continuous",
        "valid": ["continuous"],
    }
    ATT_LOW = {"type": "float", "default": 0.0}
    ATT_HIGH = {"type": "float", "default": 1.0}
    ATT_IS_PERIODIC = {"type": "bool", "default": False}

    def __str__(self):
        return f"Continuous (name='{self.name}', low={self.low}, high={self.high}, is_periodic={self.is_periodic})"

    def _validate(self):
        return self.low < self.high



class ObjectParameterCategorical(ObjectParameter):

    ATT_TYPE = {
        "type": "string",
        "default": "categorical",
        "valid": ["categorical"],
    }
    ATT_OPTIONS = {"type": "list", "default": []}
    ATT_DESCRIPTORS = {"type": "list", "default": []}

    def __str__(self):
        return f"Categorical (name='{self.name}', num_opts: {len(self.options)}, options={self.options}, descriptors={self.descriptors})"

    def _validate(self):
        return (
            len(self.options) == len(self.descriptors)
            if len(self.descriptors) > 0
            else len(self.options) > 0
        )

    @property
    def volume(self):
        return len(self.options)


class ObjectParameterDiscrete(ObjectParameter):

    ATT_TYPE = {"type": "string", "default": "discrete", "valid": ["discrete"]}
    ATT_LOW = {"type": "float", "default": 0.0}
    ATT_HIGH = {"type": "float", "default": 1.0}
    ATT_STRIDE = {"type": "float", "default": 0.1}
    ATT_OPTIONS = {"type": "list", "default": []}

    def __str__(self):
        return f"Discrete (name='{self.name}', low={self.low}, high={self.high}, stride={self.stride}, options={self.options})"

    def __contains__(self, val):
        contains = isinstance(val, int)
        contains = contains and self.low <= val <= self.high
        contains = contains and (val - self.low) % self.stride == 0
        return contains

    def _validate(self):
        if len(self.options)!=0:
            # if we have some options, forget about the stride parameter
            # just make sure options are in order
            # reset bounds based on the options
            self.low = np.amin(self.options)
            self.high = np.amax(self.options)
            return self.options == sorted(self.options)
            
        else:
            # we have use the stride parameter set by the user
            return all(
                [
                    self.low < self.high,
                    *[
                        isinstance(_, float)
                        for _ in (self.low, self.high, self.stride)
                    ],
                ]
            )

    @property
    def volume(self):
        return self.high - self.low


class ObjectParameterOrdinal(ObjectParameter):

    ''' Ordinal parameters are similar to categorical parameters, except that they
    feature a clear ordering between the options. NOTE: Olympus currently only
    supports ordinal parameters as target objectives.

    Conventionally, when instantiating an ordinal parameter in Olympus we list the
    options from least optimal to most optimal from left to right. For instance, assume
    one was conducting an experiment targeting the formation of large crystals and the
    experiment outcome options were "large_crystals", "fine_powder", "no_crystals"
    and "small_crystals". One would define the ordinal parameter as

    value0 = ParameterOrdinal(
        name="my_value"
        options=["no_crystals", "fine_powder", "small_crystals", "large_crystals"]
    )
    '''

    ATT_TYPE = {
        "type": "string",
        "default": "ordinal",
        "valid": ["ordinal"],
    }
    ATT_OPTIONS = {"type": "list", "default": []}

    def __str__(self):
        return f"Ordinal (name='{self.name}', num_opts={len(self.options)}, options={self.options}, order={np.arange(len(self.options))})"

    def _validate(self):
        # TODO: this is a hack
        return len(self.options) >= 0


    @property
    def volume(self):
        return len(self.options)



class Parameter(ObjectParameter):

    KINDS = {
        "continuous": ObjectParameterContinuous,
        "ordinal": ObjectParameterOrdinal,
    }

    def __init__(self, kind="continuous", **kwargs):
        if kind in self.KINDS:
            self.kind = kind
            for prop in dir(self.KINDS[kind]):
                if prop.startswith("ATT_"):
                    setattr(self, prop, getattr(self.KINDS[kind], prop))
            self.KINDS[kind].__init__(self)
            for key, value in kwargs.items():
                if "ATT_{}".format(key.upper()) in dir(self):
                    self.add(key, value)
            if not self.KINDS[kind]._validate(self):
                message = "Could not validate {}".format(str(self))
                Logger.log(message, "WARNING")
        else:
            message = """Could not initialize parameter.
            Parameter kind {} is unknown. Please choose from {}""".format(
                kind, ",".join(list(self.KINDS.keys()))
            )
            Logger.log(message, "ERROR")


    def __str__(self):
        return self.KINDS[self.kind].__str__(self)



# ======================================================================

# DEBUGGING

if __name__ == "__main__":

    # continuous params
    param_0 = ObjectParameterContinuous(name="param_0")
    print(param_0.__str__())
    print("=" * 50)

    # discrete params
    param_1 = ObjectParameterDiscrete(name="param_1")
    print(param_1.__str__())
    print("=" * 50)

    # categorcial params
    param_2 = ObjectParameterCategorical(name="param_2")
    print(param_2.__str__())
    print("=" * 50)
