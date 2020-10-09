.. _dataset_hplc:

HPLC N9
=======

This dataset reports the peak response of an automated high-performance liquid
chromatography (HPLC) system for varying process parameters.

The dataset includes 1,386 samples with six parameters and one objective

=================== ========== ================ ====================================
Feature             Kind       Settings         Description
=================== ========== ================ ====================================
sample_loop         continuous [ 0.00,  0.08]   volume of the sample loop [ml]
additional_volume   continuous [ 0.00,  0.06]   volume required to draw sample [ml]
tubing_volume       continuous [ 0.1,   0.9]    volume required to drive sample [ml]
sample_flow         continuous [ 0.5,   2.5]    draw rate of sample pump [ml/min]
push_speed          continuous [80,   150]      draw rate of push pump [Hz]
wait_time           continuous [ 1,    10]      wait time  [s]
=================== ========== ================ ====================================


================= ========== ========
Objective         Kind       Goal
================= ========== ========
Peak area         continuous maximize
================= ========== ========

The dataset can be downloaded from:
https://github.com/aspuru-guzik-group/phoenics

.. rubric:: Reference

.. [#f1] L.M. Roch*, F. Häse*, C. Kreisbeck, T. Tamayo-Mendoza, L.P.E. Yunker, J.E. Hein, A. Aspuru-Guzik. ChemOS: an orchestration software to democratize autonomous discovery. chemRxiv preprint, (2018), 10.26434/chemrxiv.5953606.v1.
