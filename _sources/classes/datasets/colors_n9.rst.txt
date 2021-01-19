.. _dataset_colors_n9:

Colors N9
=========

This dataset consists of colors prepared by mixing varying amounts of 3 colored dyes (red, green, blue).
The parameters represent fractions of each dye used in a mixture. The target is the normalized green-like RGB value
[0.16, 0.56, 0.28]. This dataset was collected with an N9 robotic arm from North Robotics. [#f1]_

The dataset includes 102 samples with four parameters and one objective.

=============== ========== ============== ========================================
Feature         Kind       Settings       Description
=============== ========== ============== ========================================
red             continuous [ 0, 1]        amount of red
green           continuous [ 0, 1]        amount of green
blue            continuous [ 0, 1]        amount of blue
=============== ========== ============== ========================================

================= ========== ========
Objective         Kind       Goal
================= ========== ========
distance to green continuous minimize
================= ========== ========

.. rubric:: Reference

.. [#f1] L.M. Roch, F. Häse, C. Kreisbeck, T. Tamayo-Mendoza, L.P.E. Yunker, J.E. Hein, A. Aspuru-Guzik. ChemOS: Orchestrating autonomous experimentation. Science Robotics (2018), 3(19), eaat5559.