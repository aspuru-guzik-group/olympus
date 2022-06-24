#!/usr/bin/env 

'''
Licensed to the Apache Software Foundation (ASF) under one or more 
contributor license agreements. See the NOTICE file distributed with this 
work for additional information regarding copyright ownership. The ASF 
licenses this file to you under the Apache License, Version 2.0 (the 
"License"); you may not use this file except in compliance with the 
License. You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT 
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the 
License for the specific language governing permissions and limitations 
under the License.

The code in this file was developed at Harvard University (2018) and 
modified at ChemOS Inc. (2019) as stated in the NOTICE file.
'''

__author__  = 'Florian Hase'

#=========================================================================

import sys

from utilities import PhoenicsModuleError, PhoenicsVersionError

#========================================================================

try:
    import tensorflow as tf
except ModuleNotFoundError:
    _, error_message, _ = sys.exc_info()
    extension = '\n\tTry installing the tensorflow package or use a different backend instead.\n\tTo install tensorflow execute:\n\n\tpip install tensorflow==1.4.1'
    PhoenicsModuleError(str(error_message) + extension)

if not tf.__version__ in ['1.4.0', '1.4.1']:
    PhoenicsVersionError('cannot operate with tensorflow version: "%s".\n\tPlease install version 1.4.1\n\tTo install tensorflow execute:\n\n\tpip install tensorflow==1.4.1' % tf.__version__)

try: 
    import edward as ed
except ModuleNotFoundError:
    _, error_message, _ = sys.exc_info()
    extension = '\n\tTry installing the edward package or use a different backend instead.\n\tTo install edward execute:\n\n\tpip install edward==1.3.5'
    PhoenicsModuleError(str(error_message) + extension)

if not ed.__version__ in ['1.3.5']:
    PhoenicsVersionError('cannot operate with edward version: "%s".\n\tPlease install version 1.3.5\n\tTo install edward execute:\n\n\tpip install edward==1.3.5' % ed.__version__)

#========================================================================


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

from BayesianNetwork.EdwardInterface.numpy_graph      import NumpyGraph
from BayesianNetwork.EdwardInterface.edward_interface import EdwardNetwork

