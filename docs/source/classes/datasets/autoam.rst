.. _dataset_autoam:

AutoAM
=======

The ``autoam`` dataset reports the result of an autonomous optimization of four continuous-valued 3d printing pa-
rameters to optimize the geometry of the leading segment of printed lines to target specifications.14 The objective is
termed the “shape score”, which measures the similarity between the printed line and the target specifications, and
should be maximized. [#f1]_

=================== ========== ============== ========================================
Feature             Kind       Settings       Description
=================== ========== ============== ========================================
prime delay         continuous [0.0 − 5.0]    delay before deposition commencement [s] 
print speed         continuous [0.1 − 10.0]   deposition rate [mm s−1] 
x offset correction continuous [−1.0 − 1.0]   x-component of offset vector [mm] 
y offset correction continuous [−1.0 − 1.0]   y-component of offset vector [mm] 
=================== ========== ============== ========================================

================= ========== ========
Objective         Kind       Goal
================= ========== ========
shape score       continuous maximize
================= ========== ========

.. rubric:: Reference

.. [#f1] J. R. Deneault, J. Chang, J. Myung, D. Hooper, A. Armstrong, M. Pitt, and B. Maruyama. Toward autonomous additive manufacturing: Bayesian optimization on a 3d printer. MRS Bulletin. (2021), 1–10.