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

The datasets currently available are the following:

.. toctree::
   :maxdepth: 1

   alkox
   fullerenes
   hplc_n9
   photobleaching_pce10
   photobleaching_wf3
   snar

===== =========================== ============== ================= ======
 No.   Dataset                    Kind Keyword   Objective Name    Goal
===== =========================== ============== ================= ======
  1    :ref:`dataset_alkox`       alkox          reaction rate     Max
  2    color mixing (bob)         colormix_bob   green-ness        Max
  3    color mixing (n9)          colormix_n9    green-ness        Max
  4    excitonics                 excitonics     efficiency        Max
  5    :ref:`dataset_fullerenes`  fullerenes     yield of X3       Max
  6    :ref:`dataset_hplc`        hplc           peak area         Max
  7    :ref:`dataset_photo_pce10` photobl        stability         Max
  8    :ref:`dataset_photo_wf3`   photobl        stability         Max
  9    :ref:`dataset_snar`        snar           e_factor          Max
 10    n_benzylation              ??             e_factor         Max
 11    suzuki                     suzuki         %AY
 12    ptc                        ??             TBD
 13    ada thin film?             ??             pseudo-mobility   Max
===== =========================== ============== ================= ======

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
