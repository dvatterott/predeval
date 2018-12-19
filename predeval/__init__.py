# -*- coding: utf-8 -*-
"""
    predeval
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    A library for comparing a model's output to the expected output from the model.
    :copyright: (c) 2019 by Dan Vatterott
    :license: MIT, see LICENSE for more details.
"""

__version__ = '0.0.1'
__license__ = 'MIT'

from .continuous import ContinuousEvaluator

__all__ = ['ContinuousEvaluator']
