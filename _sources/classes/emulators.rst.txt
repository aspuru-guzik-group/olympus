.. _emulators:

Emulators
=========

**Olympus** provides many pre-trained emulators to be readily used as benchmarks for optimization and experimental
planing algorithms. In particular, any combination of :ref:`datasets` and :ref:`models` specifies an ``Emulator``
that you can readily load from **Olympus** as follows::

    # we want to load the emulator that uses a Bayesian neural network model for the HPLC dataset
    from olympus import Emulator
    emulator = Emulator(dataset='hplc_n9', model='BayesNeuralNet')

Once an ``Emulator`` instance has been created, it can be used to simulate the outcome of an experimental evaluation
of the query parameters::

    next_point = [[0.01, 0.02, 0.5, 1.0, 100., 7.]]
    emulator.run(next_point)
    >>>> [ParamVector(peak_area = 244.39515369060487)]

In addition to loading pre-trained emulators, you can define your own custom emulators by creating custom instances of
``Dataset`` and ``Model``. For example, if you wanted to train an ``Emulator`` using different settings for the
``BayesNeuralNet`` model::

    from olympus import Emulator
    from olympus.models import BayesNeuralNet

    model = BayesNeuralNet(hidden_depth=3, hidden_nodes=48, out_act='sigmoid')
    emulator = Emulator(dataset='hplc_n9', model=model)

You can then evaluate the performance on the model via cross validation::

    emulator.cross_validate()

And then finally train the model::

    emulator.train()

The same can be done for a custom dataset. In this case you would load your own dataset (see :ref:`datasets`) and train
the emulator::

    from olympus import Emulator, Dataset
    from olympus.models import BayesNeuralNet

    mydata = pd.from_csv('mydata.csv')
    dataset = Dataset(data=mydata)
    model = BayesNeuralNet(hidden_depth=3, hidden_nodes=48, out_act='sigmoid')
    emulator = Emulator(dataset=dataset, model=model)
    emulator.train()

To save the ``Emulator`` instance to file, such that you do not have to re-train it every time you'd like to use it,
you can use the ``save`` method::

    emulator.save('my_new_emulator')

You can then retrieve this emulator with the ``load_emulator`` function::

    from olympus.emulators import load_emulator
    emulator = load_emulator('my_new_emulator')


Emulator Class
--------------

.. currentmodule:: olympus.emulators.emulator

.. autoclass:: Emulator
   :noindex:
   :exclude-members: add, from_dict, get, to_dict


   .. rubric:: Methods

   .. autosummary::
      run
      train
      save
      cross_validate

