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

You can explore ``Olympus`` using the following Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aspuru-guzik-group/olympus/blob/master/olympus_get_started.ipynb)

### Dependencies
The installation only requires:
* ``python >= 3.6``
* ``numpy``
* ``pandas``

Additional libraries are required to use specific modules and objects. ``Olympus`` will alert you about these requirements as you try access the related functionality.

###  Citation
``Olympus`` is research software. If you make use of it in scientific publications, please cite [the following article](https://dx.doi.org/10.1088/2632-2153/abedc8):

```bib
@article{hase_olympus_2021,
  title = {Olympus: A Benchmarking Framework for Noisy Optimization and Experiment Planning},
  shorttitle = {Olympus},
  author = {Häse, Florian and Aldeghi, Matteo and Hickman, Riley J. and Roch, Loïc M. and Christensen, Melodie and Liles, Elena and Hein, Jason E. and {Aspuru-Guzik}, Alán},
  year = {2021},
  month = jul,
  journal = {Mach. Learn.: Sci. Technol.},
  volume = {2},
  number = {3},
  pages = {035021},
  publisher = {{IOP Publishing}},
  issn = {2632-2153},
  doi = {10.1088/2632-2153/abedc8},
  abstract = {Research challenges encountered across science, engineering, and economics can frequently be formulated as optimization tasks. In chemistry and materials science, recent growth in laboratory digitization and automation has sparked interest in optimization-guided autonomous discovery and closed-loop experimentation. Experiment planning strategies based on off-the-shelf optimization algorithms can be employed in fully autonomous research platforms to achieve desired experimentation goals with the minimum number of trials. However, the experiment planning strategy that is most suitable to a scientific discovery task is a priori unknown while rigorous comparisons of different strategies are highly time and resource demanding. As optimization algorithms are typically benchmarked on low-dimensional synthetic functions, it is unclear how their performance would translate to noisy, higher-dimensional experimental tasks encountered in chemistry and materials science. We introduce Olympus, a software package that provides a consistent and easy-to-use framework for benchmarking optimization algorithms against realistic experiments emulated via probabilistic deep-learning models. Olympus includes a collection of experimentally derived benchmark sets from chemistry and materials science and a suite of experiment planning strategies that can be easily accessed via a user-friendly Python interface. Furthermore, Olympus facilitates the integration, testing, and sharing of custom algorithms and user-defined datasets. In brief, Olympus mitigates the barriers associated with benchmarking optimization algorithms on realistic experimental scenarios, promoting data sharing and the creation of a standard framework for evaluating the performance of experiment planning strategies.},
}
```
The preprint is also available at https://arxiv.org/abs/2010.04153.

###  License
``Olympus`` is distributed under an MIT License.

