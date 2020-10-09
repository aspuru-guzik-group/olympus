#!/usr/bin/env python

#======================================================================

from olympus import Logger

from olympus.objects import Object

#======================================================================

class ObjectParameter(Object):

    ATT_NAME = {'type': 'string', 'default': 'parameter'}

#======================================================================

class ObjectParameterContinuous(ObjectParameter):

    ATT_TYPE        = {'type': 'string', 'default': 'continuous', 'valid': ['continuous']}
    ATT_LOW         = {'type': 'float',  'default': 0.0}
    ATT_HIGH        = {'type': 'float',  'default': 1.0}
    ATT_IS_PERIODIC = {'type': 'bool',   'default': False}

    def __str__(self):
        return f"Continuous (name='{self.name}', low={self.low}, high={self.high}, is_periodic={self.is_periodic})"

    def _validate(self):
        return self.low < self.high

#======================================================================

class Parameter(ObjectParameter):

    KINDS = {'continuous': ObjectParameterContinuous}

    def __init__(self, kind = 'continuous', **kwargs):
        if kind in self.KINDS:
            self.kind = kind
            for prop in dir(self.KINDS[kind]):
                if prop.startswith('ATT_'):
                    setattr(self, prop, getattr(self.KINDS[kind], prop))
            self.KINDS[kind].__init__(self)
            for key, value in kwargs.items():
                if 'ATT_{}'.format(key.upper()) in dir(self):
                    self.add(key, value)
            if not self.KINDS[kind]._validate(self):
                message = 'Could not validate {}'.format(str(self))
                Logger.log(message, 'WARNING')
        else:
            message = '''Could not initialize parameter.
            Parameter kind {} is unknown. Please choose from {}'''.format(kind, ','.join(list(self.KINDS.keys())))
            Logger.log(message, 'ERROR')


    def __str__(self):
        return self.KINDS[self.kind].__str__(self)

#======================================================================
