#!/usr/bin/env python

#===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from olympus.plotter import AbstractPlotter

#===============================================================================

class PlotterMatplotlib(AbstractPlotter):

    def _plot(self, emulators, planners, measurements, file_name = None, show = False):
        num_plots  = len(emulators)
        num_graphs = len(planners)
        fig = plt.figure(figsize = (6, 4 * num_plots))
        axs = []
        for plot_index in range(num_plots):
            ax = plt.subplot2grid((num_plots, 1), (plot_index, 0))
            axs.append(ax)

        for plot_index, emulator in enumerate(emulators):
            ax = axs[plot_index]
            for planner in planners:

                idxs, vals = [], []
                indices = measurements[emulator][planner]['idxs'].copy()
                values  = measurements[emulator][planner]['vals'].copy()
                while len(indices) > 1:
                    idxs.append(indices[0])
                    vals.append(values[0])
                    indices = indices[1:]
                    values  = values[1:]
                    if idxs[-1] > indices[0]:
                        vals = np.squeeze(vals)
                        ax.plot(idxs, vals)
                        idxs, vals = [], []
                vals = np.squeeze(vals)
                ax.plot(idxs, vals)
                idxs, vals = [], []

            ax.set_title(f'{emulator.capitalize()}')

            ax.set_xlabel('# evaluations')
            ax.set_ylabel('measurement')

        plt.tight_layout()

        if file_name is not None:
            fig.savefig(file_name, bbox_inches = 'tight')
        if show is True:
            plt.show()
