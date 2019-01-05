#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `predeval` package."""
import sys
import os
import numpy as np
# import pytest

sys.path.append(os.path.abspath("../predeval"))
from predeval import ContinuousEvaluator  # noqa pylint: disable=W0611, C0413
from predeval import CategoricalEvaluator  # noqa pylint: disable=W0611, C0413


class TestContinuous(object):
    """Class containing continuous evaluator tests."""

    con_eval = ContinuousEvaluator(np.array([x for x in range(31)]))

    def test_inheritance(self):
        """Assert that continuous evaluator inheriting from parent."""
        assert 'check_data' in dir(self.con_eval), 'Inheritance failed'

    def test_min_param(self):
        """Assert that correctly setting min value."""
        assert self.con_eval.assertion_params['minimum'] == 0

    def test_max_param(self):
        """Assert that correctly setting max value."""
        assert self.con_eval.assertion_params['maximum'] == 30

    def test_mean_param(self):
        """Assert that correctly setting mean value."""
        assert self.con_eval.assertion_params['mean'] == 15.0

    def test_std_param(self):
        """Assert that correctly setting std value."""
        assert self.con_eval.assertion_params['std'] == 8.9442719099991592

    def test_kstest_param(self):
        """Assert that correctly setting ks_stat."""
        assert self.con_eval.assertion_params['ks_stat'] == 0.2

    def test_kstest(self, capsys):
        """Assert that correctly ks_test."""
        self.con_eval.check_ks(np.array([x for x in range(51)]))
        captured = capsys.readouterr()
        assert captured.out == "Failed ks check; test statistic=0.3922, p=0.0036\n"

    def test_update_param(self):
        """Assert that correctly updating parameters."""
        self.con_eval.update_param('minimum', -1)
        assert self.con_eval.assertion_params['minimum'] == -1
