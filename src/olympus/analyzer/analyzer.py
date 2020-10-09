#!/usr/bin/env python

import numpy as np

from olympus import Object

# ==============================================================================


class Analyzer(Object):
    def __init__(self, campaigns=[]):
        Object.__init__(**locals())

    def _get_best_vals(self, campaigns):
        vals_ = [campaign.best_values for campaign in campaigns]
        max_len = np.amax([len(val_) for val_ in vals_])
        vals = np.zeros((len(vals_), max_len)) + np.nan
        for _, val_ in enumerate(vals_):
            vals[_, : len(val_)] = np.squeeze(val_[:, 0])
        return vals

    def set_campaigns(self, campaigns):
        self.campaigns = campaigns

    def _get_best_reduction(self, operator, campaigns=None, locs=None, ci_method=None, ci_size=100):
        ''' computes statistics on the provided campaigns

        Statistics are computed following the specific reduction operation. Uncertainty
        estimates can be requested by choosing a method to determine confidence
        intervals and specifying the desired interval range (0-100)

        Args:
            operator (): reduction operator
            campaigns (list of campaign): campaigns for which to compute statistics
            locs (array of floats): locations at which statistics should be computed
            ci_method (str): method to estimate uncertainties (choose from: None, "bootstrap")
            ci_size (int): size of confidence interval (0-100)

        Returns:
            array: computed statistics
        '''
        if campaigns is None:
            campaigns = self.campaigns
        best_vals = self._get_best_vals(campaigns)
        if locs is None:
            best_vals_red = operator(best_vals, axis=0)
        else:
            locs = np.array(locs)
            locs = locs[np.where(locs < best_vals.shape[1])[0]]
            best_vals_red = operator(best_vals, axis=0)[locs - 1]

        if ci_method is None:
            return best_vals_red

        if ci_method == 'bootstrap':

            # implementing bootstrapping
            NUM_BOOTS = 100

            # samples = (# num_boot, # campaigns, length of trace)
            idxs    = np.random.randint(low=0, high=best_vals.shape[0], size=(NUM_BOOTS, best_vals.shape[0], best_vals.shape[1]))
            samples = []
            for _ in range(NUM_BOOTS):
                sample = [best_vals[idxs[_, :, __], __] for __ in range(best_vals.shape[1])]
                samples.append(sample)
            samples = np.array(samples)
            reductions = operator(samples, axis=2)

            lower_reductions = np.empty(best_vals.shape[1])
            upper_reductions = np.empty(best_vals.shape[1])

            for _ in range(best_vals.shape[1]):
                sorted_reductions = np.sort(reductions[:, _])
                lower_reductions[_] = sorted_reductions[int( (100 - ci_size) / 2 * (NUM_BOOTS-1) / 100)]
                upper_reductions[_] = sorted_reductions[int( (ci_size - (100 - ci_size) / 2) * (NUM_BOOTS-1) /  100)]

            return best_vals_red, {'lower': lower_reductions, 'upper': upper_reductions}

        else:
            raise NotImplementedError


    def get_best_mean(self, campaigns=None, locs=None, ci_method=None, ci_size=100):
        return self._get_best_reduction(np.nanmean, campaigns, locs, ci_method=ci_method, ci_size=ci_size)

    def get_best_median(self, campaigns=None, locs=None):
        return self._get_best_reduction(np.nanmedian, campaigns, locs)

    def get_best_std(self, campaigns=None, locs=None):
        return self._get_best_reduction(np.nanstd, campaigns, locs)

    def get_best_min(self, campaigns=None, locs=None):
        return self._get_best_reduction(np.nanmin, campaigns, locs)

    def get_best_max(self, campaigns=None, locs=None):
        return self._get_best_reduction(np.nanmax, campaigns, locs)
