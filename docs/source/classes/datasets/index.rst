.. _datasets:

Datasets
========

**Olympus** provides various datasets form across the natural sciences that form the basis of realistic and challenging
benchmarks for optimization algorithms. :ref:`models` trained on these datasets provide :ref:`emulators` that are used
to simulate an experimental campaign.

While you can load pre-trained :ref:`emulators` based on these datasets, you can load these datasets with ``Dataset``
class::

    from olympus.datasets import Dataset
    dataset = Dataset(kind='snar')


Fully categorical datasets do not use emualtors. Rather, given a certain set of parameters, objective values are looked up in a table. For fully categorical datasets (e.g., ``redoxmers``, ``dye_lasers``, ``suzuki_edbo``), the ``Dataset`` instance itself funtions in a similar way to an emulator. Given a set of parameters, one can retrive the objective function value(s) as follows::

   from olympus.objects import ParameterVector
   from olympus.datasets import Dataset



The datasets currently available are the following:

.. toctree::
   :maxdepth: 1

   alkox
   colors_bob
   colors_n9
   fullerenes
   hplc
   benzylation
   photo_pce10
   photo_wf3
   snar
   suzuki
   diffvap_crystal
   dye_lasers
   redoxmers
   perovskites
   oer_plate_a
   oer_plate_b
   oer_plate_c
   oer_plate_d
   p3ht
   agnp
   thin_films
   crossed_barrel
   autoam
   suzuki_i
   suzuki_ii
   suzuki_iii
   suzuki_iv
   suzuki_edbo
   buchwald_a
   buchwald_b
   buchwald_c
   buchwald_d
   buchwald_e

===== ===============================  ===============  ========================================== ==============
 No.   Dataset                         Kind Keyword     Objectives                                 Goals
===== ===============================  ===============  ========================================== ==============
  1    :ref:`dataset_alkox`            alkox            reaction rate                              Max
  2    :ref:`dataset_colors_bob`       colors_bob       green-ness                                 Min
  3    :ref:`dataset_colors_n9`        colors_n9        green-ness                                 Min
  4    :ref:`dataset_fullerenes`       fullerenes       yield of X1+X2                             Max
  5    :ref:`dataset_hplc`             hplc             peak area                                  Max
  6    :ref:`dataset_photo_pce10`      photo_pce10      stability                                  Min
  7    :ref:`dataset_photo_wf3`        photo_wf3        stability                                  Min
  8    :ref:`dataset_snar`             snar             e_factor                                   Min
  9    :ref:`dataset_benzylation`      benzylation      e_factor                                   Min
 10    :ref:`dataset_suzuki`           suzuki           yield                                      Max
 11    :ref:`dataset_diffvap_crystal`  diffvap_crystal  crystal_score                              Max
 12    :ref:`dataset_dye_lasers`       dye_lasers       peak_score, spectral_overlap, fluo_rate    Max, Min, Max
 13    :ref:`dataset_redoxmers`        redoxmers        abs_lam_diff, ered, gsol                   Min, Min, Min
 14    :ref:`dataset_perovskites`      perovskites      hse_gap                                    Min
 15    :ref:`dataset_oer_plate_a`      oer_plate_a      overpotential                              Min
 16    :ref:`dataset_oer_plate_b`      oer_plate_b      overpotential                              Min
 17    :ref:`dataset_oer_plate_c`      oer_plate_c      overpotential                              Min
 18    :ref:`dataset_oer_plate_d`      oer_plate_d      overpotential                              Min
 19    :ref:`dataset_p3ht`             p3ht             conductivity                               Max
 20    :ref:`dataset_agnp`             agnp             spectrum_score                             Max
 21    :ref:`dataset_thin_films`       thin_films       instability_index                          Min
 22    :ref:`dataset_crossed_barrel`   crossed_barrel   toughness                                  Max
 23    :ref:`dataset_autoam`           autoam           shape_score                                Max
 24    :ref:`dataset_suzuki_i`         suzuki_i         yield, turnover                            Max, Max
 25    :ref:`dataset_suzuki_ii`        suzuki_ii        yield, turnover                            Max, Max
 26    :ref:`dataset_suzuki_iii`       suzuki_iii       yield, turnover                            Max, Max
 27    :ref:`dataset_suzuki_iv`        suzuki_iv        yield, turnover                            Max, Max
 28    :ref:`dataset_suzuki_edbo`      suzuki_edbo      yield                                      Max
 29    :ref:`buchwald_a`               buchwald_a       yield                                      Max
 30    :ref:`buchwald_b`               buchwald_b       yield                                      Max
 31    :ref:`buchwald_c`               buchwald_c       yield                                      Max
 32    :ref:`buchwald_d`               buchwald_d       yield                                      Max
 33    :ref:`buchwald_e`               buchwald_e       yield                                      Max
===== ===============================  ===============  ========================================== ==============

In addition to the **Olympus** datasets, you can load your own custom ones::

    from olympus.datasets import Dataset
    import pandas as pd

    mydata = pd.from_csv('mydata.csv')
    dataset = Dataset(data=mydata)

Dataset Class
-------------

.. currentmodule:: olympus.datasets

.. autoclass:: Dataset
   :noindex:
   :exclude-members: add, from_dict, generate, get, to_dict


   .. rubric:: Methods

   .. autosummary::
      dataset_info
      set_param_space
      get_cv_fold
