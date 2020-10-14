Overview
========

The main components of **Olympus** and their relationships are depicted in the image below. The core classes that most
users will find useful are ``Planner``, ``Emulator``, and ``Surface``. ``Model`` and ``Dataset`` can be used to create
new, custom ``Emulator`` objects.

A ``Campaign`` instance can be used to store all information related to a specific optimization, from the results to the
details of the algorithms used. An ``Evaluator`` can store the details of multiple optimization campaigns.

.. image:: _static/core-classes-map.png