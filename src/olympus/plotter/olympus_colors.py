#!/usr/bin/env python

# =====================
# Define Olympus colors
# =====================

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

_olympus_reference_colors = [
    "#08294C",
    "#75BBE1",
    "#D4E9F4",
    "#F2F2F2",
    "#F7A4D4",
    "#F75BB6",
    "#EB0789",
]
_olympus_cmap = LinearSegmentedColormap.from_list(
    "olympus", _olympus_reference_colors
)
_olympus_cmap_r = LinearSegmentedColormap.from_list(
    "olympus_r", _olympus_reference_colors[::-1]
)

mpl.colormaps.register(cmap=_olympus_cmap)
mpl.colormaps.register(cmap=_olympus_cmap_r)


def get_olympus_colors(n):
    olympus_cmap = mpl.colormaps.get_cmap("olympus")
    return [olympus_cmap(x) for x in np.linspace(0, 1, n)]
