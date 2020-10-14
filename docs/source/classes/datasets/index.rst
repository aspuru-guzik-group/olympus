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
   colors_bob
   colors_n9
   fullerenes
   hplc
   benzylation
   photo_pce10
   photo_wf3
   snar
   suzuki


===== =========================== ============== ================= ======
 No.   Dataset                    Kind Keyword   Objective         Goal
===== =========================== ============== ================= ======
  1    :ref:`dataset_alkox`       alkox          reaction rate     Max
  2    :ref:`dataset_colors_bob`  colors_bob     green-ness        Min
  3    :ref:`dataset_colors_n9`   colors_n9      green-ness        Min
  4    :ref:`dataset_fullerenes`  fullerenes     yield of X1+X2    Max
  5    :ref:`dataset_hplc`        hplc           peak area         Max
  6    :ref:`dataset_photo_pce10` photo_pce10    stability         Min
  7    :ref:`dataset_photo_wf3`   photo_wf3      stability         Min
  8    :ref:`dataset_snar`        snar           e_factor          Min
  9    :ref:`dataset_benzylation` benzylation    e_factor          Min
 10    :ref:`dataset_suzuki`      suzuki         yield             Max
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
