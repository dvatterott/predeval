Welcome to predeval's documentation!
======================================

This software is built to identify unexpected changes in a model output before evaluation data becomes available. For example, if you create a churn model, you will have to wait X number of weeks before learning whether users churned (and can evaluate your churn model predictions). This software will not guarantee that your model is accurate, but it will alert you if your model's outputs (i.e., predictions) are dramatically different from what they have been in the past.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   readme
   installation
   usage
   api
   contributing
   authors
   history

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
