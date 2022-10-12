## Olympus: a benchmarking framework for noisy optimization and experiment planning
[![Build Status](https://travis-ci.com/FlorianHase/olympus.svg?token=bMWWqBdm3xytautMLsPK&branch=dev)](https://travis-ci.com/FlorianHase/olympus)
[![codecov](https://codecov.io/gh/FlorianHase/olympus/branch/flo/graph/badge.svg?token=FyvePgBDQ5)](https://codecov.io/gh/FlorianHase/olympus)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


![alt text](https://github.com/aspuru-guzik-group/olympus/blob/dev/docs/source/_static/logo2b.png)


``Olympus`` provides a consistent and easy-to-use **framework for benchmarking optimization algorithms**. With ``olympus`` you can:
* Build optimization domains using **continuous**, **discrete** and **categorical** parameter types.
* Access a suite of **23 experiment planning algortihms** via a simple and consistent interface
* Access **33 experimentally-derived benchmarks** and **33 analytical test functions** for optimization benchmarks
* Easily integrate custom optimization algorithms
* Easily integrate custom datasets, which can be used to train models for custom benchmarks
* Enjoy extensive plotting and analysis options for visualizing your benchmark experiments

You can find more details in the [documentation](https://aspuru-guzik-group.github.io/olympus/).

###  Installation
``Olympus`` can be installed with ``pip``:

```
pip install olymp
```

The package can also be installed via ``conda``:

```
conda install -c conda-forge olymp
```

Finally, the package can be built from source:

``` 
git clone https://github.com/aspuru-guzik-group/olympus.git
cd olympus
python setup.py develop
```

You can explore ``Olympus`` using the following Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aspuru-guzik-group/olympus/blob/master/olympus_get_started.ipynb)

### Dependencies
The installation only requires:
* ``python >= 3.6``
* ``numpy``
* ``pandas``

Additional libraries are required to use specific modules and objects. ``Olympus`` will alert you about these requirements as you try access the related functionality.

### Use cases
The following projects have used ``Olympus`` to streamline the benchmarking of optimization algorithms.

* [Bayesian optimization with known experimental and design constraints for chemistry applications](https://pubs.rsc.org/en/content/articlelanding/2022/dd/d2dd00028h)
* [Golem: an algorithm for robust experiment and process optimization](https://pubs.rsc.org/en/content/articlelanding/2021/sc/d1sc01545a)
* [Equipping data-driven experiment planning for Self-driving Laboratories with semantic memory: case studies of transfer learning in chemical reaction optimization](https://chemrxiv.org/engage/chemrxiv/article-details/6276f20987d01f0f03dcbe10)




###  Citation
``Olympus`` is research software. If you make use of it in scientific publications, please cite the following article:

```
@article{hase_olympus_2021,
      author = {H{\"a}se, Florian and Aldeghi, Matteo and Hickman, Riley J. and Roch, Lo{\"\i}c M. and Christensen, Melodie and Liles, Elena and Hein, Jason E. and Aspuru-Guzik, Al{\'a}n},
      doi = {10.1088/2632-2153/abedc8},
      issn = {2632-2153},
      journal = {Machine Learning: Science and Technology},
      month = jul,
      number = {3},
      pages = {035021},
      title = {Olympus: a benchmarking framework for noisy optimization and experiment planning},
      volume = {2},
      year = {2021}
}
```

###  License
``Olympus`` is distributed under an MIT License.

