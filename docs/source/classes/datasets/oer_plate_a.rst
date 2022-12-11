.. _dataset_oer_plate_a:

OER Plate A
============

The oer plate datasets comprise 4 high-thoughput screens for oxygen evolution reaction (OER) activity by systematically exploring high-dimensional chemical spaces.7–9 The four datasets each contain a discrete library of 2121 catalysts, comprising all unary, binary, ternary and quaternary compositions from unique 6 element sets with 10 at% intervals. The composition systems for each of the four datasets are as follows, a) Mn-Fe-Co-Ni-La-Ce, b) Mn-Fe-Co-Ni-Cu-Ta, c) Mn-Fe-Co-Cu-Sn-Ta, and d) Ca-Mn-Co-Ni-Sn-Sb. During the optimizations, experiment planning strategies traverse the entire standard 6-simplex of catalyst compositions. Stein et al. [#f1]_  report only unary, binary, ternary and quaternary compositions from 6 element sets with 10 at% intervals. For the compositions whose overpo-
tentials are not reported in the original dataset, i.e., the quinary and senary compositions, a probabilistic emulator is used to produce a virtual measurement. We do not claim that these extrapolated values are quantitatively accurate with respect to experiment, only that they are reasonable values for overpotentials with respect to measured values. The oer plate datasets feature an additional constraint that valid parameters must be on the 6-simplex. To enforce this, we allow the experiment planner to operate on the standard 5-cube, and map proposals to the 6-simplex using
a deterministic transformation before they are processed by the emulator. [#f1]_ [#f2]_ [#f3]_



=============== ========== ============== ==================================================
Feature         Kind       Settings       Description
=============== ========== ============== ==================================================
mat 1           continuous [ 0.0, 1.0]    fractional composition of system material 1 [a.u.]
mat 2           continuous [ 0.0, 1.0]    fractional composition of system material 2 [a.u.]
mat 3           continuous [ 0.0, 1.0]    fractional composition of system material 3 [a.u.]
mat 4           continuous [ 0.0, 1.0]    fractional composition of system material 4 [a.u.]
mat 5           continuous [ 0.0, 1.0]    fractional composition of system material 5 [a.u.]
mat 6           continuous [ 0.0, 1.0]    fractional composition of system material 6 [a.u.]
=============== ========== ============== ==================================================

================= ========== ========
Objective         Kind       Goal
================= ========== ========
overpotential     continuous minimize
================= ========== ========

.. rubric:: Reference

.. [#f1] E. Soedarmadji, H. S. Stein, S. K. Suram, D. Guevarra, and J. M. Gregoire. Tracking materials science data lineage to manage millions of materials experiments and analyses. npj Computational Materials 5 (2019), 1–9.

.. [#f2] B. Rohr, H. S. Stein, D. Guevarra, Y. Wang, J. A. Haber, M. Aykol, S. K. Suram, and J. M. Gregoire. Benchmarking the acceleration of materials discovery by sequential learning. Chemical Science. 11 (2020), 2696–2706.

.. [#f3] H. S. Stein, D. Guevarra, A. Shinde, R. J. R. Jones, J. M. Gregoire, and J. A. Haber,. Functional mapping reveals mechanistic clusters for OER catalysis across (Cu–Mn–Ta–Co–Sn–Fe)Ox composition and pH space. Materials Horizons. 6 (2019), 1251–1258.