#!/usr/bin/env python

# ===============================================================================

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_context("paper", font_scale=1.5)

from olympus.plotter import AbstractPlotter

# ===============================================================================


class PlotterSeaborn(AbstractPlotter):
    def _set_color_palette(self, line_theme="deep"):
        # NOTE/WARNING: the number of individual elements in this color palette
        # should be increased when more planners are added ...
        self.line_palette = sns.color_palette(line_theme, 20)

    def _plot_traces(
        self,
        emulators,
        planners,
        measurements,
        file_name=None,
        show=False,
        *args,
        **kwargs,
    ):

        self._set_color_palette()
        num_plots = len(emulators)
        num_graphs = len(planners)
        fig = plt.figure(figsize=(6, 5 * num_plots))
        axs = []
        for plot_index in range(num_plots):
            ax = plt.subplot2grid((num_plots, 1), (plot_index, 0))
            axs.append(ax)

        for plot_index, emulator in enumerate(emulators):
            ax = axs[plot_index]
            for planner_ix, planner in enumerate(planners):
                measurements[emulator][planner]["vals"] = np.squeeze(
                    measurements[emulator][planner]["vals"]
                )
                sns.lineplot(
                    x="idxs",
                    y="vals",
                    data=measurements[emulator][planner],
                    color="k",
                    ax=ax,
                    linewidth=5,
                    ci=None,
                )
                sns.lineplot(
                    x="idxs",
                    y="vals",
                    data=measurements[emulator][planner],
                    ax=ax,
                    linewidth=4,
                    color=self.line_palette[planner_ix],
                    label=planner,
                )
            ax.grid(linestyle=":")
            ax.set_title(f"{emulator.capitalize()}")

            ax.set_xlabel("# evaluations")
            ax.set_ylabel("measurement")

        # plt.legend(loc='upper right', fontsize=12)
        plt.legend(fontsize=12)
        plt.tight_layout()

        if file_name is not None:
            plt.savefig(file_name, bbox_inches="tight")
        if show is True:
            plt.show()

    def _plot_traces_regret(
        self,
        emulators,
        planners,
        measurements,
        file_name=None,
        show=False,
        *args,
        **kwargs,
    ):

        self._set_color_palette()
        num_plots = len(emulators)
        num_graphs = len(planners)
        fig = plt.figure(figsize=(6, 5 * num_plots))
        axs = []
        for plot_index in range(num_plots):
            ax = plt.subplot2grid((num_plots, 1), (plot_index, 0))
            axs.append(ax)

        for plot_index, emulator in enumerate(emulators):
            ax = axs[plot_index]
            for planner_ix, planner in enumerate(planners):
                measurements[emulator][planner]["vals"] = np.squeeze(
                    measurements[emulator][planner]["vals"]
                )
                sns.lineplot(
                    x="idxs",
                    y="vals",
                    data=measurements[emulator][planner],
                    color="k",
                    ax=ax,
                    linewidth=5,
                    ci=None,
                )
                sns.lineplot(
                    x="idxs",
                    y="vals",
                    data=measurements[emulator][planner],
                    ax=ax,
                    linewidth=4,
                    color=self.line_palette[planner_ix],
                    label=planner,
                )
            ax.grid(linestyle=":")
            ax.set_title(f"{emulator.capitalize()}")

            ax.set_xlabel("# evaluations")
            ax.set_ylabel("regret")

        # plt.legend(loc='upper right', fontsize=12)
        plt.legend(fontsize=12)
        plt.yscale("log")
        plt.tight_layout()

        if file_name is not None:
            plt.savefig(file_name, bbox_inches="tight")
        if show is True:
            plt.show()

    def _plot_traces_rank(
        self,
        emulators,
        planners,
        measurements,
        file_name=None,
        show=False,
        *args,
        **kwargs,
    ):

        self._set_color_palette()
        num_plots = len(emulators)
        num_graphs = len(planners)
        fig = plt.figure(figsize=(6, 5 * num_plots))
        axs = []
        for plot_index in range(num_plots):
            ax = plt.subplot2grid((num_plots, 1), (plot_index, 0))
            axs.append(ax)

        for plot_index, emulator in enumerate(emulators):
            ax = axs[plot_index]
            for planner_ix, planner in enumerate(planners):
                measurements[emulator][planner]["vals"] = np.squeeze(
                    measurements[emulator][planner]["vals"]
                )
                sns.lineplot(
                    x="idxs",
                    y="vals",
                    data=measurements[emulator][planner],
                    color="k",
                    ax=ax,
                    linewidth=5,
                    ci=None,
                )
                sns.lineplot(
                    x="idxs",
                    y="vals",
                    data=measurements[emulator][planner],
                    ax=ax,
                    linewidth=4,
                    color=self.line_palette[planner_ix],
                    label=planner,
                )
            ax.grid(linestyle=":")
            ax.set_title(f"{emulator.capitalize()}")

            ax.set_xlabel("# evaluations")
            ax.set_ylabel("best candidate rank")

        # plt.legend(loc='upper right', fontsize=12)
        plt.legend(fontsize=12)
        plt.yscale("log")
        plt.tight_layout()

        if file_name is not None:
            plt.savefig(file_name, bbox_inches="tight")
        if show is True:
            plt.show()

    def _plot_traces_fraction_top_k(
        self,
        emulators,
        planners,
        measurements,
        threshold,
        is_percent,
        file_name=None,
        show=False,
        *args,
        **kwargs,
    ):

        self._set_color_palette()
        num_plots = len(emulators)
        num_graphs = len(planners)
        fig = plt.figure(figsize=(6, 5 * num_plots))
        axs = []
        for plot_index in range(num_plots):
            ax = plt.subplot2grid((num_plots, 1), (plot_index, 0))
            axs.append(ax)

        for plot_index, emulator in enumerate(emulators):
            ax = axs[plot_index]
            for planner_ix, planner in enumerate(planners):
                measurements[emulator][planner]["vals"] = np.squeeze(
                    measurements[emulator][planner]["vals"]
                )
                sns.lineplot(
                    x="idxs",
                    y="vals",
                    data=measurements[emulator][planner],
                    color="k",
                    ax=ax,
                    linewidth=5,
                    ci=None,
                )
                sns.lineplot(
                    x="idxs",
                    y="vals",
                    data=measurements[emulator][planner],
                    ax=ax,
                    linewidth=4,
                    color=self.line_palette[planner_ix],
                    label=planner,
                )
            ax.grid(linestyle=":")
            ax.set_title(f"{emulator.capitalize()}")

            ax.set_xlabel("# evaluations")
            if not is_percent:
                ax.set_ylabel(f"fraction top-{threshold} candidates")
            else:
                ax.set_ylabel(f"fraction top-{threshold}\npercent candidates")

        plt.legend(loc="upper left", fontsize=12)
        plt.tight_layout()

        if file_name is not None:
            plt.savefig(file_name, bbox_inches="tight")
        if show is True:
            plt.show()

    def _plot_num_evals_top_k(
        self,
        emulators,
        planners,
        measurements,
        threshold,
        is_percent,
        file_name=None,
        show=False,
        *args,
        **kwargs,
    ):

        self._set_color_palette()
        num_plots = len(emulators)
        num_graphs = len(planners)
        fig = plt.figure(figsize=(6, 5 * num_plots))
        axs = []
        for plot_index in range(num_plots):
            ax = plt.subplot2grid((num_plots, 1), (plot_index, 0))
            axs.append(ax)

        for plot_index, emulator in enumerate(emulators):
            ax = axs[plot_index]
            all_data = {"planner": [], "vals": []}
            for planner_ix, planner in enumerate(planners):
                # measurements[emulator][planner]['vals'] = np.squeeze(measurements[emulator][planner]['vals'])
                all_data["planner"].extend(
                    measurements[emulator][planner]["planner"]
                )
                all_data["vals"].extend(
                    measurements[emulator][planner]["vals"]
                )
            sns.boxplot(
                x="planner",
                y="vals",
                # hue='planner',
                data=all_data,
                # color=self.line_palette[:len(planners)],
                ax=ax,
                linewidth=2.0,
            )
            sns.swarmplot(
                x="planner",
                y="vals",
                # hue='planner',
                data=all_data,
                # color=self.line_palette[:len(planners)],
                ax=ax,
                linewidth=0.5,
            )

            ax.grid(linestyle=":")
            ax.set_title(f"{emulator.capitalize()}")

            ax.set_xlabel("planner")
            if not is_percent:
                ax.set_ylabel(f"# evaluations for top-{threshold}\ncandidate")
            else:
                ax.set_ylabel(
                    f"# evaluations for top-{threshold}\npercent candidates"
                )

        plt.tight_layout()

        if file_name is not None:
            plt.savefig(file_name, bbox_inches="tight")
        if show is True:
            plt.show()

    def _plot_regret_x_evals(
        self,
        emulators,
        planners,
        measurements,
        num_evals,
        is_cumulative,
        file_name=None,
        show=False,
        *args,
        **kwargs,
    ):

        self._set_color_palette()
        num_plots = len(emulators)
        num_graphs = len(planners)
        fig = plt.figure(figsize=(6, 5 * num_plots))
        axs = []
        for plot_index in range(num_plots):
            ax = plt.subplot2grid((num_plots, 1), (plot_index, 0))
            axs.append(ax)

        for plot_index, emulator in enumerate(emulators):
            ax = axs[plot_index]
            all_data = {"planner": [], "vals": []}
            for planner_ix, planner in enumerate(planners):
                # measurements[emulator][planner]['vals'] = np.squeeze(measurements[emulator][planner]['vals'])
                all_data["planner"].extend(
                    measurements[emulator][planner]["planner"]
                )
                all_data["vals"].extend(
                    measurements[emulator][planner]["vals"]
                )
            sns.boxplot(
                x="planner",
                y="vals",
                # hue='planner',
                data=all_data,
                # color=self.line_palette[:len(planners)],
                ax=ax,
                linewidth=2.0,
            )
            sns.swarmplot(
                x="planner",
                y="vals",
                # hue='planner',
                data=all_data,
                # color=self.line_palette[:len(planners)],
                ax=ax,
                linewidth=0.5,
            )

            ax.grid(linestyle=":")
            ax.set_title(f"{emulator.capitalize()}")

            ax.set_xlabel("planner")
            if is_cumulative:
                ax.set_ylabel(
                    f"cumulative regret after\n{num_evals} iterations"
                )
            else:
                ax.set_ylabel(f"regret aft3er\n{num_evals} iterations")

        plt.tight_layout()

        if file_name is not None:
            plt.savefig(file_name, bbox_inches="tight")
        if show is True:
            plt.show()
