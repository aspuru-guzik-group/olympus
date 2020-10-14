.. _dataset_photo_wf3:

Photobleaching WF3
==================


This dataset reports the degradation of polymer blends for organic solar cells
under the exposure to light. Individual data points encode the ratios of
individual polymers in one blend, along with the measured photodegradation of
this blend. [#f1]_

The dataset includes 1,040 samples with four parameters and one objective.

======= ========== ======== ================
Feature Kind       Settings Description
======= ========== ======== ================
mat_1   continuous [0, 1]   amount of WF3
mat_2   continuous [0, 1]   amount of P3HT
mat_3   continuous [0, 1]   amount of PCBM
mat_4   continuous [0, 1]   amount of oIDTBR
======= ========== ======== ================


================= ========== ========
Objective         Kind       Goal
================= ========== ========
photo-degradation continuous minimize
================= ========== ========

The dataset can be downloaded from:
https://github.com/aspuru-guzik-group/quaterny_opvs


.. rubric:: Reference

.. [#f1] S. Langner*, F. Häse*, J.D. Perea, T. Stubhan, J. Hauch, L.M. Roch, T. Heumueller, A. Aspuru-Guzik, C.J. Brabec. Beyond Ternary OPV: High-Throughput Experimentation and Self-Driving Laboratories Optimize Multicomponent Systems. Advanced Materials, 2020, 1907801.
