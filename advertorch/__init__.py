# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import os

with open(os.path.join(os.path.dirname(__file__), 'VERSION')) as f:
    __version__ = f.read().strip()

from . import attacks  # noqa: F401
from . import defenses  # noqa: F401

"""
    Comments by Hanshu YAN, ECE, NUS
    - attacks, various types of attack methods included for robustness evaluation
    - defenses, various types of preprocessing methods included for image transformation
"""
