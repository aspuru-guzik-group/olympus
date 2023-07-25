#!/usr/bin/env python

from abc import abstractmethod

from olympus.objects import Object


class AbstractModel(Object):
    def __init__(self, *args, **kwargs):
        Object.__init__(self, *args, **kwargs)

    @abstractmethod
    def train(self, X, y):
        """Method to train the model. We expect some standard output from this method."""
        pass

    @abstractmethod
    def predict(self, X):
        """Method that returns a prediction. We expect some standard output from this method."""
        pass
