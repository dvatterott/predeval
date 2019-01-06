========
predeval
========


.. image:: https://img.shields.io/pypi/v/predeval.svg
        :target: https://pypi.python.org/pypi/predeval

.. image:: https://img.shields.io/travis/dvatterott/predeval.svg
        :target: https://travis-ci.org/dvatterott/predeval

.. image:: https://readthedocs.org/projects/predeval/badge/?version=latest
        :target: https://predeval.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://codecov.io/gh/dvatterott/predeval/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/dvatterott/predeval

.. image:: https://pyup.io/repos/github/dvatterott/predeval/shield.svg
     :target: https://pyup.io/repos/github/dvatterott/predeval/
     :alt: Updates



This software is built to identify unexpected changes in a model output before evaluation data becomes available. For example, if you create a churn model, you will have to wait X number of weeks before learning whether users churned (and can evaluate your churn model predictions). This software will not guarantee that your model is accurate, but it will alert you if your model's outputs (i.e., predictions) are dramatically different from what they have been in the past.


* Free software: MIT license
* Documentation: https://predeval.readthedocs.io.


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
