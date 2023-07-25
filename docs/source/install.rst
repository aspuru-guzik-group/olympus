Installation
============

We recommend to install **Olympus** with ``pip``::

    pip install olymp

Alternatively, you can install **Olympus** with ``conda``::

    conda install -c conda-forge  olymp

Finally, you can clone the GitHub repo and install it from source::

    git clone git@github.com:aspuru-guzik-group/olympus.git
    cd olympus
    python setup.py install


Dependencies
------------
The installation only requires:

* ``python >= 3.6``
* ``numpy``
* ``pandas``

Additional libraries are required to use specific :ref:`planners` and :ref:`emulators`. However, **Olympus** will alert
you about these requirements as you try access the related functionality.




