#!/usr/bin/env python

from abc import abstractmethod

import numpy as np
import pandas as pd

# # TODO: make this import dependent on which ASF strategy we are using
#from chimera import Chimera

from olympus import Logger
from olympus.objects import Object, ABCMeta


class AbstractASF(Object, metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        Object.__init__(self, *args, **kwargs)


    @abstractmethod
    def scalarize(self, objectives):
        pass

    @abstractmethod
    def validate_asf_params(self):
        pass



class ChimeraASF(AbstractASF):
    ''' The Chimera achievement scalarizing function.
    Chem. Sci., 2018, 9, 7642
    '''
    def __init__(self, value_space, tolerances, absolutes, goals):
        AbstractASF.__init__(**locals())

        from chimera import Chimera

        self.validate_asf_params()
        self.chimera = Chimera(
            tolerances=self.tolerances, absolutes=self.absolutes, goals=self.goals
        )


    def scalarize(self, objectives):
        ''' this expects a (# obs, # objs) numpy array, which is scalarized
        according to the given tolerances and goals. Returns a (# obs,)
        numpy array corresponding to the merits of each observation, 0 corresponding
        to the best value, and 1 corresponding to the worst value
        '''
        assert len(objectives.shape)==2
        return self.chimera.scalarize(objectives)

    def validate_asf_params(self):

        if not (len(self.tolerances)==len(self.absolutes)==len(self.goals)):
            message = 'Lengths of Chimera parameters do not match'
            Logger.log(message, 'FATAL')
        if not len(self.tolerances)==len(self.value_space):
            message = 'Number of Chimera parameters does not match the number of objectives'
            Logger.log(message, 'FATAL')


class WeightedSumASF(AbstractASF):
    ''' simple weighted sum acheivement scalarizing function
    weights is a 1d numpy array of
    '''
    def __init__(self, value_space, weights, goals):
        AbstractASF.__init__(**locals())

        self.validate_asf_params()
        # normalize the weight values such that their magnitudes
        # sum to 1
        self.norm_weights = self.softmax(self.weights)
        self.norm_weights = [weight if self.goals[idx]=='min' else -weight for idx, weight in enumerate(self.norm_weights)]

    def scalarize(self, objectives):
        norm_objectives = self.normalize(objectives)
        merit = np.sum(norm_objectives*self.norm_weights, axis=1)
        # final normalization
        # smaller merit values are best
        merit = self.normalize(merit)
        return merit

    @staticmethod
    def softmax(vector):
        vector = vector/np.amax(weights)
        return np.exp(weights) / np.sum(np.exp(weights))

    @staticmethod
    def normalize(vector):
        min_ = np.amin(vector)
        max_ = np.amax(vector)
        ixs = np.where(np.abs(max_-min_)<1e-10)[0]
        if not ixs.size == 0:
            max_[ixs]=np.ones_like(ixs)
            min_[ixs]=np.zeros_like(ixs)
        return (vector - min_) / (max_ - min_)

    def validate_asf_params(self):
        if not np.all(np.array(self.weights)>=0.):
            message = 'Weighted sum ASF weights must be non-negative real numbers'
            Logger.log(message, 'FATAL')
        if not len(self.weights)==len(self.value_space):
            message = 'Number of weights does not match the number of objectives'
            Logger.log(message, 'FATAL')


class ConstrainedASF(AbstractASF):

    def __init__(self, value_space, lowers, uppers, delta_fs):
        AbstractASF.__init__(**locals())

        self.validate_asf_params()


    def scalarize(self, objectives):
        return None

    def validate_asf_params(self):
        if not (len(self.lowers)==len(self.uppers)==len(self.delta_fs)):
            message = 'c-ASF parameters not the same length'
            Logger.log(message, 'FATAL')
        if not len(self.lowers) == len(self.value_space):
            message = 'Number of c-ASF parameters do not match the number of objectives'
            Logger.log(message, 'FATAL')
