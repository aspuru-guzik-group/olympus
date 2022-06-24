''' Phoenics

'''
from setuptools import dist
dist.Distribution().fetch_build_eggs(['numpy'])

import numpy as np

from setuptools import setup
from distutils.extension import Extension

#===============================================================================

def readme():
    with open('README.md', encoding="utf-8") as content:
        return content.read()

#===============================================================================

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass    = {}
ext_modules = []

if use_cython:
    ext_modules += [
        Extension('phoenics.BayesianNetwork.kernel_evaluations', ['src/phoenics/BayesianNetwork/kernel_evaluations.pyx']),
    ]
    cmdclass.update({'build_ext': build_ext})
else:
    ext_modules += [
        Extension('phoenics.BayesianNetwork.kernel_evaluations', ['src/phoenics/BayesianNetwork/kernel_evaluations.c']),
    ]

#===============================================================================

setup(
    name             = 'phoenics',
    version          = '0.2.0',
    description      = 'Phoenics: A deep Bayesian optimizer',
    long_description = readme(),
	long_description_content_type = 'text/markdown',
    classifiers      = [
        'Intended Audience :: Science/Research',
		'Operating System :: Unix',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
    ],
    url              = 'https://github.com/chemos-inc/phoenics',
    author           = 'ChemOS Inc.',
    author_email     = 'florian@chemos.io',
    license          = 'Apache license, version 2',
    packages         = [
		'phoenics',
		'phoenics/Acquisition',
		'phoenics/Acquisition/NumpyOptimizers',
		'phoenics/BayesianNetwork',
		'phoenics/BayesianNetwork/EdwardInterface',
		'phoenics/BayesianNetwork/TfprobInterface',
        'phoenics/DatabaseHandler',
        'phoenics/DatabaseHandler/JsonWriters',
        'phoenics/DatabaseHandler/PandasWriters',
        'phoenics/DatabaseHandler/PickleWriters',
        'phoenics/DatabaseHandler/SqliteInterface',
		'phoenics/ObservationProcessor',
		'phoenics/RandomSampler',
		'phoenics/SampleSelector',
		'phoenics/utilities',
	],
    package_dir      = {'': 'src'},
    cmdclass         = cmdclass,
    ext_modules      = ext_modules,
    include_dirs     = np.get_include(),
    zip_safe         = False,
    tests_require    = ['pytest'],
    install_requires = [
		'numpy',
		'pyyaml>=5.1',
		'sqlalchemy>=1.3',
		'watchdog>=0.9',
		'wheel>=0.33'
	],
    python_requires  = '>=3.6',
)
