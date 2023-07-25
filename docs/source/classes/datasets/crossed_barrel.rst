.. _dataset_crossed_barrel:

Crossed Barrel
==============

The ``crossed_barrel`` dataset reports the results from a data-driven optimization of 3D printed parts for their
mechanical properties (in this case, their toughness). A platform which combines additive manufacturing, robotics,
and mechanical testing was employed. The system prints a crossed barrel family of structures, which are supported by
n hollow columns with outer radius r and thickness t, twisted at an angle θ. After printing, the structures are subjected
to uni-axial compression. The toughness objective is then recorded as the area under the resulting force-displacement
curve, and is intended to be maximized. [#f1]_

=============== ========== ============== ========================================
Feature         Kind       Settings       Description
=============== ========== ============== ========================================
n               discrete   [6, 12]        number of hollow columns
theta           continuous [0.0, 200.0]   twist angle of columns [degrees]
r               continuous [1.5, 2.5]     outer radius of the columns [mm]
t               continuous [0.7 − 1.4]    thickness of the hollow columns [mm]
=============== ========== ============== ========================================

================= ========== ========
Objective         Kind       Goal
================= ========== ========
toughness         continuous maximize
================= ========== ========

.. rubric:: Reference

.. [#f1] A. E. Gongora, B. Xu, W. Perry, C. Okoye, P. Riley, K. G. Reyes, E. F. Morgan, and K. A. Brown. A bayesian experimental autonomous researcher for mechanical design. Science Advances. 6 (2020), eaaz1708.