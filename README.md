## Olympus: a benchmarking framework for noisy optimization and experiment planning
[![Build Status](https://travis-ci.com/FlorianHase/olympus.svg?token=bMWWqBdm3xytautMLsPK&branch=dev)](https://travis-ci.com/FlorianHase/olympus)
[![codecov](https://codecov.io/gh/FlorianHase/olympus/branch/flo/graph/badge.svg?token=FyvePgBDQ5)](https://codecov.io/gh/FlorianHase/olympus)

``Olympus`` provides a consistent and easy-to-use **framework for benchmarking optimization algorithms**. With ``olympus`` you can:
* Access a suite of **18 experiment planning algortihms** via a simple and consistent interface
* Easily integrate custom optimization algorithms
* Access **10 experimentally-derived benchmarks** emulated with probabilistic models, and **23 analytical test functions** for optimization
* Easily integrate custom datasets, which can be used to train models for custom benchmarks

You can find more details in the [documentation](https://aspuru-guzik-group.github.io/olympus/).

###  Installation
``Olympus`` can be installed with ``pip``:

```
pip install olymp
```

### Dependencies
The installation only requires:
* ``python >= 3.6``
* ``numpy``
* ``pandas``

Additional libraries are required to use specific modules and objects. ``Olympus`` will alert you about these requirements as you try access the related functionality.

###  Citation
``Olympus`` is research software. If you make use of it in scientific publications, please cite the following article:

```
@misc{olympus,
      title={Olympus: a benchmarking framework for noisy optimization and experiment planning}, 
      author={Florian Häse and Matteo Aldeghi and Riley J. Hickman and Loïc M. Roch and Melodie Christensen and Elena Liles and Jason E. Hein and Alán Aspuru-Guzik},
      year={2020},
      eprint={2010.04153},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```

###  License
``Olympus`` is distributed under an MIT License.

