.. _dataset_thin_films:

Thin Films
==========

The ``thin_films`` dataset reports the results of a closed-loop machine-learning driven optimization of the stability
of lead iodide perovskite materials that suffer from heat- and moisture-induced degradation.12 The material search
space is the five-element space CsxMAy FA1−x−y PbI3. Thin-film samples are spin-coated before being examined under
85% relative humidity and 85◦C. The objective of this dataset is minimization of the perovskite material’s instability
index, which is defined as the integrated color change of the films over the accelerated degradation period [#f1]_

=============== ========== ============== ========================================
Feature         Kind       Settings       Description
=============== ========== ============== ========================================
CsPbI           continuous [0.0 − 1.0]    fractional composition of CsPbI
FAPbI           continuous [0.0 − 1.0]    fractional composition of FAPbI
MAPbI           continuous [0.0 − 1.0]    fractional composition of MAPbI
=============== ========== ============== ========================================

================= ========== ========
Objective         Kind       Goal
================= ========== ========
instability index continuous minimize
================= ========== ========

.. rubric:: Reference

.. [#f1]  S. Sun, A. Tiihonen, F. Oviedo, Z. Liu, J. Thapa, Y. Zhao, N. T. P. Hartono, A. Goyal, T. Heumueller, C. Batali, et al. A data fusion approach to optimize compositional stability of halide perovskites. Matter 4 (2021), 1305–1322.