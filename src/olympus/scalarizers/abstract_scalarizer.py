#!/usr/bin/env python

from abc import abstractmethod

from olympus.objects import Object, ABCMeta


class AbstractScalarizer(Object, metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        Object.__init__(self, *args, **kwargs)


    @abstractmethod
    def scalarize(self, objectives):
        pass

    @abstractmethod
    def validate_asf_params(self):
        pass

