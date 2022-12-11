.. _dataset_dye_lasers:

Dye Lasers
===========

The dye lasers dataset reports computed photophysical properties for 3458 organic molecules synthesized from three groups of molecular building blocks – A, B, and C (resulting in A-B-C-B-A pentamers).4 Three syntheses are used: iterative Suzuki-Miyaura cross-couling reactions, nucleophilic aromatic substitutions and Buchwald-Hartwig animations. Each molecule was subjected to a computational protocol consisting of cheminformatic, semi-empirical and ab initio quantum chemical steps to compute absorption and emission spectra and fluorescence rates. The objectives of this dataset, in order of decreasing importance are i) the peak score, which is a dimensionless quantity given by the fraction of the fluorescence power spectral density that falls within the 400 − 460 nm region, ii) the spectral overlap of the absorption and emission spectra, and iii) the fluorescence rate. [#f1]_


=============== =========== ============== ========================================
Feature         Kind        Settings       Description
=============== =========== ============== ========================================
A fragment      categorical 14             terminal fragment 
B fragment      categorical 13             bridge fragment 
C fragment      categorical 19             core fragment 
=============== =========== ============== ========================================

================= ========== ========
Objective         Kind       Goal
================= ========== ========
peak score        continuous maximize
spectral overlap  continuous minimize
fluorescence rate continuous maximize
================= ========== ========

.. rubric:: Reference

.. [#f1] M. Seifrid, R. J. Hickman, A. Aguilar-Granda, C. Lavigne, J. Vestfrid, T. C. Wu, T. Gaudin, E. J. Hopkins, and A. Aspuru-Guzik. Routescore: Punching the ticket to more efficient materials development. ACS Central Science, 8 (2022) 122–131.