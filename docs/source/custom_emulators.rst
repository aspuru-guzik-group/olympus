Custom Emulators
================
You can create your own emulator, based on a custom dataset within **Olympus**. A simple example where we create an
``Emulator`` for Boston housing dataset can be found in the examples. Here we provide a high-level guide of the steps
necessary.

First, you need to create a custom ``Dataset``. This can easily be done if you have your data in a ``DataFrame``. This
needs to contain all parameters and measurements/targets, such that you can specify the column name for the target of interest ::

   from olympus.datasets import Dataset
   import pandas as pd

   df = pd.from_csv('mydata.csv')
   dataset = Dataset(data=df, target_ids=['target'])

Second, you need to specify the parameter space of the problem. An easy of doing this is to ask **Olympus** to infer this
from the data ::

   dataset.infer_param_space()

However, to have more control over this you can instantiate and populate a ``ParameterSpace`` object to then pass to
``dataset``::

   from olympus import ParameterSpace, Parameter

   # initialise a parameter space object
   param_space = ParameterSpace()

   # define the parameters and append them to param_space
   param1 = Parameter(kind='continuous', name='param1', low=10, high=100)
   param2 = Parameter(kind='continuous', name='param2', low=5, high=15)
   param_space.add(param1)
   param_space.add(param2)

   # provide this param_space to dataset
   dataset.set_param_space(param_space)

Now the details of your data are all set. So as third step, you need to instantiate the ``Model`` that will be trained
on this data::

   from olympus.models import BayesNeuralNet

   model = BayesNeuralNet(hidden_depth=2, hidden_nodes=12, hidden_act='leaky_relu', out_act="relu",
                          batch_size=50, reg=0.005, max_epochs=10000)


Finally, you combine dataset and model into an ``Emulator``::

   from olympus import Emulator
   emulator = Emulator(dataset=dataset, model=model, feature_transform='normalize', target_transform='normalize')

The choice of ``feature_transform``, ``target_transform``, as well as the model hyperparameters depend on the details
of your dataset. Once all this is setup, you can use cross-validation to test your ``emulator`` and tweak the model's
architecture::

   emulator.cross_validate()

Once you are happy with the performance of the emulator, you can train it::

  emulator.train()

After this, you have a custom emulator ready to be used to benchmark various optimization algorithms for your specific
problem::

   emulator.run(...)

You can also save this specific emulator to be reused::

   # save
   emulator.save('custom_emulator')

   # load
   emulator = load_emulator('custom_emulator')