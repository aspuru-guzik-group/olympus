.. _dataset_alkox:

Alkoxylation
============

This dataset reports the biocatalytic oxidation of benzyl alcohol by a copper radical oxidase (AlkOx). The effects of
enzyme loading, cocatalyst loading, and pH balance on both initial rate and total conversion were assayed. [#f1]_

The dataset includes 104 samples with four parameters and one objective.

=============== ========== ============== ========================================
Feature         Kind       Settings       Description
=============== ========== ============== ========================================
Catalase        continuous [ 0.05,  1.0]  concentration [μM]
Peroxidase      continuous [  0.5, 10.0]  concentration [μM]
Alcohol oxidase continuous [  2.0,  8.0]  concentration [nM]
pH              continuous [    6,    8]  -log(H+)
=============== ========== ============== ========================================

================= ========== ========
Objective         Kind       Goal
================= ========== ========
conversion        continuous maximize
================= ========== ========

.. rubric:: Reference

.. [#f1] F. Häse, M. Aldeghi, R.J. Hickman, L.M. Roch, M. Christensen, E. Liles, J.E. Hein, A. Aspuru-Guzik. Olympus: a benchmarking framework for noisy optimization and experiment planning. arXiv (2020), 2010.04153.