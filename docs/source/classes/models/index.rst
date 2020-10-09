.. _models:

Models
======

**Olympus** provides easy access to different models that can be trained to emulate an experiment. When loading an
``Emulator`` with a standard dataset and model, one of the models provided will be loaded from file. However, if you
would like to customize the model used by an ``Emulator``, you can do this with one of the model classes. Similar to
:ref:`planners`, you can either import the desired class directly::

    from olympus.models import BayesNeuralNet
    model = BayesNeuralNet(...)

Or you can take advantage of the ``Model`` function if you want to use the class with default arguments::

    from olympus.models import Model
    model = Model(kind='BayesNeuralNet')

The models available are the following:

.. toctree::
   :maxdepth: 1

   bnn
   nn