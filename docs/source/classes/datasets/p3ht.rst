.. _dataset_p3ht:

P3HT
=====

The ``p3ht`` dataset reports the electrical conductivity of composite thin films prepared using a machine learning-driven automated flow mixing setup with a high-throughput drop-casting system. [#f1]_ Regio-regular poly-3-hexylthiophene (rr-P3HT) is combined with 4 types of carbon nanotubes (CNTs), leading to to different morphologies and crystaline structures which modulate the electrical conductivity of the thin film. The types of CNTs used in the study are i) long single wall CNTs (l-SWNTs, 5-30 μm), ii) short single wall CNTs (s-SWNTs, 1-3 μm), iii) multi-walled CNTs
(MWCNTs), and iv) double-walled CNTs (MWCNTs). The films are processed by optical and electrical diagnostics to asses their electrical conductivity, which is meant to be maximized. 

=============== ========== ============== ========================================
Feature         Kind       Settings       Description
=============== ========== ============== ========================================
p3ht content    continuous [15.0, 96.27]  rr-P3HT polymer content
d1 content      continuous [0.0, 60.0]    l-SWNT carbon nanotube content
d2 content      continuous [0.0, 70.0]    s-SWNT carbon nanotube content
d6 content      continuous [0.0, 85.0]    MWCNT carbon nanotube content
d8 content      continuous [0.0, 75.0]    DWCNT carbon nanotube content
=============== ========== ============== ========================================

================= ========== ========
Objective         Kind       Goal
================= ========== ========
conductivity      continuous maximize
================= ========== ========

.. rubric:: Reference

.. [#f1]  D. Bash, Y. Cai, V. Chellappan, S. L. Wong, X. Yang, P. Kumar, J. D. Tan, A. Abutaha, J. J. Cheng, Y.-F. Lim, et al. Multi-fidelity high-throughput optimization of electrical conductivity in p3ht-cnt composites. Advanced Functional Materials (2021), 2102606.