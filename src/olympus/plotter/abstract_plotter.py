#!/usr/bin/env python

#===============================================================================

import abc
import numpy as np

#===============================================================================

from olympus import Object

#===============================================================================

class AbstractPlotter(Object):


    @abc.abstractmethod
    def _plot(self, emulators, planners, measurements, *args, **kwargs):
        pass


    def plot_from_db(self, database, emulator=None, plot_file_name='test.png', *args, **kwargs):
        campaigns    = database.get_campaigns()
        emulators    = []
        planners     = []
        measurements = {}
        for campaign in campaigns:
            if campaign.emulator_type == 'n/a':  # this means campaign never got an Emulator or Surface
                continue
            emulator_type = campaign.get_emulator_type()
            if emulator_type == 'numeric':
                camp_emulator = f'{campaign.model_kind}_{campaign.dataset_kind}'
            elif emulator_type == 'analytic':
                camp_emulator = f'analytic_{campaign.surface_kind}'
            if emulator is None:
                emulator = camp_emulator
            if camp_emulator != emulator: continue
            planner  = campaign.get_planner_kind()
            if emulator not in emulators:
                emulators.append(emulator)
                measurements[emulator] = {}
            if planner not in planners:
                planners.append(planner)
                measurements[emulator][planner] = {'idxs': [], 'vals': []}
            for val_index, value in enumerate(campaign.best_values):
#                print('MEASUREMENTS', measurements[emulator].keys(), val_index, campaign.id)
#                try:
                measurements[emulator][planner]['idxs'].append(val_index)
                measurements[emulator][planner]['vals'].append(value)
#                except KeyError: continue

        for emulator in emulators:
            for planner in planners:
                try:
                    measurements[emulator][planner]['idxs'] = np.array(measurements[emulator][planner]['idxs'])
                    measurements[emulator][planner]['vals'] = np.array(measurements[emulator][planner]['vals'])
                except KeyError: continue

        self._plot(emulators, planners, measurements, file_name=plot_file_name, *args, **kwargs)
