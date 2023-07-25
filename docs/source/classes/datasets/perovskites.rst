.. _dataset_perovskites:

Perovskites
============

The perovskites dataset reports simulated bandgaps (HSE06 level of theory) for 192 hybrid organic-inorganic perovskite (HOIP) materials The HOIP candidates of this dataset are designed from a set of 4 different halide anions, 3 different group-IV cations and 16 different organic anions. Electronic and geometric descriptors of the HOIP components are also provided. We characterize the inorganic constituents (anion and cation) by their electron affinity, ionization energy, mass, and electronegativity. Organic components are described by their HOMO and LUMO energies, dipole moment, atomization energy, radius of gyration, and molecular weight. [#f1]_


=============== =========== ============== ========================================
Feature         Kind        Settings       Description
=============== =========== ============== ========================================
organic         categorical 16     		   organic anion
cation          categorical 3    		   group-IV cation
anion           categorical 4   		   halide anion
=============== =========== ============== ========================================

================= ========== ========
Objective         Kind       Goal
================= ========== ========
HSE06 bandgap     continuous minimize
================= ========== ========

.. rubric:: Reference

.. [#f1] C. Kim, T. Doan Huan, S. Krishnan, and R. Ramprasad. A hybrid organic-inorganic perovskite dataset. Scientific Data. 4 (2017), 1–11.