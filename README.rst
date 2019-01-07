========
predeval
========


.. image:: https://img.shields.io/pypi/v/predeval.svg
        :target: https://pypi.python.org/pypi/predeval

.. image:: https://img.shields.io/travis/dvatterott/predeval.svg
        :target: https://travis-ci.org/dvatterott/predeval

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/dvatterott/predeval?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/dvatterott/predeval

.. image:: https://readthedocs.org/projects/predeval/badge/?version=latest
        :target: https://predeval.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://codecov.io/gh/dvatterott/predeval/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/dvatterott/predeval

.. image:: https://pyup.io/repos/github/dvatterott/predeval/shield.svg
     :target: https://pyup.io/repos/github/dvatterott/predeval/
     :alt: Updates

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/predeval.svg
   :alt: Supported versions
   :target: https://pypi.python.org/pypi/predeval


This software is built to identify changes in a model output before evaluation data becomes available.

For example, if you create a churn model, you will have to wait X number of weeks before learning whether users churned (and can evaluate your churn model predictions).

This software will not guarantee that your model is accurate, but it will alert you if your model's outputs (i.e., predictions) are different from what they have been in the past. A model's output can pass predeval tests and be inaccurate and a model's output can fail predeval and be accurate. That said, unexpected changes in model outputs likely represent a change in accuracy.


* Free software: MIT license
* Documentation: https://predeval.readthedocs.io.

Installation
-------

Installation is described here: https://predeval.readthedocs.io/en/latest/installation.html

Example Usage
-------

Examples can be found here: https://predeval.readthedocs.io/en/latest/usage.html


API Documentation
-------

Documentation of the software can be found here: https://predeval.readthedocs.io/en/latest/api.html

Contributing
-------

Info about contributing can be found here: https://predeval.readthedocs.io/en/latest/contributing.html

Changelog
-------

Changelog can be found here: https://predeval.readthedocs.io/en/latest/history.html

Credits
-------

Info about contributors can be found here: https://predeval.readthedocs.io/en/latest/authors.html

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
