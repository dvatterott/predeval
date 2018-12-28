#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `predeval` package."""
import sys
import os
import pytest

sys.path.append(os.path.abspath("../predeval"))
from predeval import ContinuousEvaluator  # noqa pylint: disable=W0611, C0413


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string
