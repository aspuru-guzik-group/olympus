.. _dataset_suzuki:

Suzuki reaction
===============

This dataset reports palladium-catalyzed Suzuki cross-coupling between 2-bromophenyltetrazole and an electron-deficient
aryl boronate. Four reaction conditions can be controlled to maximise the reaction yield. [#f1]_

The dataset includes 247 samples with four parameters and one objective.

=============== ========== ============== ========================================
Feature         Kind       Settings       Description
=============== ========== ============== ========================================
temperature     continuous [  75,  90]     temperature of the reaction [Celsius]
Pd mol          continuous [ 0.5, 5.0]     loading of Pd catalyst [mol %]
ArBpin          continuous [ 1.0, 1.8]     equivalents of boronate ester
K3PO4           continuous [ 1.5,   3]     equivalents of tripotassium phosphate
=============== ========== ============== ========================================

================= ========== ========
Objective         Kind       Goal
================= ========== ========
yield             continuous maximize
================= ========== ========

.. rubric:: Reference

.. [#f1] F. Häse, M. Aldeghi, R.J. Hickman, L.M. Roch, M. Christensen, E. Liles, J.E. Hein, A. Aspuru-Guzik. Olympus: a benchmarking framework for noisy optimization and experiment planning. arXiv (2020), 2010.04153.