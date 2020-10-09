#!/usr/bin/env python

#===============================================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('paper', font_scale = 1.5)

from olympus.plotter import AbstractPlotter

#===============================================================================

class PlotterSeaborn(AbstractPlotter):

    def _set_color_palette(self, line_theme = 'deep'):
        # NOTE/WARNING: the number of individual elements in this color palette
        # should be increased when more planners are added ...
        self.line_palette = sns.color_palette(line_theme, 20)

    def _plot(self, emulators, planners, measurements, file_name = None, show = False):

        self._set_color_palette()
        num_plots  = len(emulators)
        num_graphs = len(planners)
        fig = plt.figure(figsize = (6, 5 * num_plots))
        axs = []
        for plot_index in range(num_plots):
            ax = plt.subplot2grid((num_plots, 1), (plot_index, 0))
            axs.append(ax)

        for plot_index, emulator in enumerate(emulators):
            ax = axs[plot_index]
            for planner_ix, planner in enumerate(planners):
                measurements[emulator][planner]['vals'] = np.squeeze(measurements[emulator][planner]['vals'])
                sns.lineplot(
                    x = 'idxs',
                    y = 'vals',
                    data = measurements[emulator][planner],
                    color = 'k',
                    ax = ax,
                    linewidth = 5,
                    ci = None
                )
                sns.lineplot(
                    x    = 'idxs',
                    y    = 'vals',
                    data = measurements[emulator][planner],
                    ax   = ax,
                    linewidth = 4,
                    color = self.line_palette[planner_ix],
                    label = planner
                )
            ax.grid(linestyle = ':')
            ax.set_title(f'{emulator.capitalize()}')

            ax.set_xlabel('# evaluations')
            ax.set_ylabel('measurement')

        #plt.legend(loc='upper right', fontsize=12)
        plt.legend(fontsize = 12)
        plt.tight_layout()

        if file_name is not None:
            plt.savefig(file_name, bbox_inches = 'tight')
        if show is True:
            plt.show()
