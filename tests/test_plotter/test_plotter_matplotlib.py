#!/usr/bin/env python

import os

from olympus import Database, Olympus
from olympus.plotter import PlotterMatplotlib as Plotter


# NOTE/WARNING: this test fails when using a few other planners, e.g. Simplex and Hyperopt, due to apparently various
# reasons. It seems there might be bugs we have not ironed out when loading things back from database (maybe
# restoring the planner status?) or other sources (out-of-bounds errors affecting param_space?)
def test_trace_plot():
    # run short campaign
    olymp = Olympus()
    olymp.run(
        planner="BasinHopping", dataset="hplc", num_iter=3
    )  # choose fast method to speed up testing

    # load database from file
    file_name = olymp.database.db.file_name
    database = Database().from_file(file_name)

    # plot campaign
    plot_file_name = "test.png"
    plotter = Plotter().plot_from_db(database, plot_file_name=plot_file_name)
    assert os.path.isfile(plot_file_name)
    # remove file
    os.remove(plot_file_name)


# ===============================================================================

if __name__ == "__main__":
    test_trace_plot()
