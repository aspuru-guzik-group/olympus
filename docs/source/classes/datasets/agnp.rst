.. _dataset_agnp:

Silver Nanoparticles
=====================

The ``agnp`` dataset is the result of an optimization of silver nanoparticles (AgNPs) for targeted absorbance spectra using a machine learning-driven high-throughput microfluidic platform. [#f1]_ The AgNP synthesis was carried out with a droplet-based platform with 5 continuous-valued parameters. Four of the parameters (QAgNO3 , QPVA, QTSC, and Qseed) are flow rate ratios, where Qi is the ratio between the flow rate of reactant i and the total aqueous flow rate. The fifth parameter, Qtotal, is the total flow rate. The objective of the optimization is the theoretical absorbance spectrum of triangular prism AgNPs with 50 nm edges and 10 nm heights as calculated by plasmon resonance simulation using discrete dipole scattering. The resulting value is termed the “spectrum score”, whose value is to be maximized.

=============== ========== ======================== ==================================================
Feature         Kind       Settings       			Description
=============== ========== ======================== ==================================================
Q AgNO3         continuous [4.53, 42.80981595]      silver nitrate flow rate ratio [%]
Q PVA           continuous [9.999518 − 40.00101474] polyvinyl alcohol flow rate ratio [%]
Q TSC           continuous [0.5 − 30.5]   			trisodium citrate flow rate ratio [%]
Q seed          continuous [0.498851653 − 19.5]     silver seed flow rate ratio [%]
Q total         continuous [200.0 − 983.0]          total (oil and aqueous phases) flow rate [μL/min]
=============== ========== ======================== ==================================================

================= ========== ========
Objective         Kind       Goal
================= ========== ========
spectrum score    continuous maximize
================= ========== ========

.. rubric:: Reference

.. [#f1] F. Mekki-Berrada, Z. Ren, T. Huang, W. K. Wong, F. Zheng, J. Xie, I. P. S. Tian, S. Jayavelu, Z. Mahfoud, D. Bash, et al. Two-step machine learning enables optimized nanoparticle synthesis. npj Computational Material. 7 (2021), 1–10.