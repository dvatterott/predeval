=====
Examples
=====

Example of using the ContinuousEvaluator

.. code:: ipython3

    from predeval import ContinuousEvaluator

    # create continuous sample.
    # this might typically be your model's output from a training data-set
    from numpy.random import uniform, seed
    seed(1234)
    model_output = uniform(0, 100, size=(1000,))

    # create evaluator object
    ce = ContinuousEvaluator(model_output)
    ce.update_param('minimum', 0)  # we know our data should not be less than 0
    ce.update_param('maximum', 100) # we also know our data should not be greater than 100

    # this might typically be your production model's output
    new_model_output = uniform(0, 100, size=(1000,))

    # check whether the new output is different than expected
    test_results = ce.check_data(new_model_output)

.. parsed-literal::

    Passed min check; min observed=0.022724991417177876
    Passed max check; max observed=99.80687158469324
    Passed mean check; mean observed=48.234379605277816 (Expected 50.8804672605415 +- 58.93838342088574)
    Passed std check; std observed=29.579104190514 (Expected 29.46919171044287 +- 14.734595855221436)
    Passed ks check; test statistic=0.051000000000000045, p=0.14408243524623565

.. code:: ipython3

    # print test outputs. note we will not generate assertion errors on failure.
    from predeval import evaluate_tests
    evaluate_tests(test_results, assert_test=False)

.. parsed-literal::

    Passed min test.
    Passed max test.
    Passed mean test.
    Passed std test.
    Passed ks test.

.. code:: ipython3

    changed_model_output = uniform(0, 100, size=(1000,)) + 20
    changed_test_results = ce.check_data(changed_model_output)

.. parsed-literal::

    Passed min check; min observed=20.004308527071295
    Failed max check; max observed=119.7728425105031
    Passed mean check; mean observed=70.78355620677603 (Expected 50.8804672605415 +- 58.93838342088574)
    Passed std check; std observed=28.94443741932546 (Expected 29.46919171044287 +- 14.734595855221436)
    Failed ks check; test statistic=0.21699999999999997, p=4.182182152969388e-21

.. code:: ipython3

    evaluate_tests(changed_test_results, assert_test=False)

.. parsed-literal::

    Passed min test.
    Failed max test.
    Passed mean test.
    Passed std test.
    Failed ks test.

Example of using the CategoricalEvaluator

.. code:: ipython3

    from predeval import CategoricalEvaluator

    # create categorical sample.
    # this might typically be your model's output from a training data-set
    from numpy.random import uniform, seed
    seed(1234)
    model_output = choice([0, 1, 2], size=(1000,))

    # create evaluator object
    ce = CategoricalEvaluator(model_output)

    # this might typically be your production model's output
    new_model_output = choice([0, 1, 2], size=(1000,))

    # check whether the new output is different than expected
    test_results = ce.check_data(new_model_output)

.. parsed-literal::

    Passed chi2 check; test statistic=0.7317191804740675, p=0.6936001826101796
    Passed min check; observed=[0 1 2] (Expected [0, 1, 2])

.. code:: ipython3

    # print test outputs. note we will not generate assertion errors on failure.
    from predeval import evaluate_tests
    evaluate_tests(test_results, assert_test=False)

.. parsed-literal::

    Passed chi2 test.
    Passed exist test.

.. code:: ipython3

    changed_model_output = choice([0, 1, 2], size=(1000,))
    changed_model_output[:200] = 0
    changed_test_results = ce.check_data(changed_model_output)

.. parsed-literal::

    Failed chi2 check; test statistic=59.06552162818124, p=1.493086411779028e-13
    Passed min check; observed=[0 1 2] (Expected [0, 1, 2])

.. code:: ipython3

    evaluate_tests(changed_test_results, assert_test=False)

.. parsed-literal::

    Failed chi2 test.
    Passed exist test.
