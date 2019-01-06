=====
Examples
=====

ContinuousEvaluator
========

Example of using the ContinuousEvaluator

.. code-block:: python3

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
    # Passed min check; min observed=0.0227
    # Passed max check; max observed=99.8069
    # Passed mean check; mean observed=48.2344 (Expected 50.8805 +- 58.9384)
    # Passed std check; std observed=29.5791 (Expected 29.4692 +- 14.7346)
    # Passed ks check; test statistic=0.0510, p=0.1441

    # print test outputs. note we will not generate assertion errors on failure.
    from predeval import evaluate_tests
    evaluate_tests(test_results, assert_test=False)
    # Passed min test.
    # Passed max test.
    # Passed mean test.
    # Passed std test.
    # Passed ks test.

    changed_model_output = uniform(0, 100, size=(1000,)) + 20
    changed_test_results = ce.check_data(changed_model_output)
    # Passed min check; min observed=20.0043
    # Failed max check; max observed=119.7728
    # Passed mean check; mean observed=70.7836 (Expected 50.8805 +- 58.9384)
    # Passed std check; std observed=28.9444 (Expected 29.4692 +- 14.7346)
    # Failed ks check; test statistic=0.2170, p=0.0000

    evaluate_tests(changed_test_results, assert_test=False)
    # Passed min test.
    # Failed max test.
    # Passed mean test.
    # Passed std test.
    # Failed ks test.

CategoricalEvaluator
========

Example of using the CategoricalEvaluator

.. code-block:: python3

    from predeval import CategoricalEvaluator

    # create categorical sample.
    # this might typically be your model's output from a training data-set
    from numpy.random import choice, seed
    seed(1234)
    model_output = choice([0, 1, 2], size=(1000,))

    # create evaluator object
    ce = CategoricalEvaluator(model_output)

    # this might typically be your production model's output
    new_model_output = choice([0, 1, 2], size=(1000,))

    # check whether the new output is different than expected
    test_results = ce.check_data(new_model_output)
    # Passed chi2 check; test statistic=0.7317, p=0.6936
    # Passed min check; observed=[0 1 2] (Expected [0, 1, 2])

    # print test outputs. note we will not generate assertion errors on failure.
    from predeval import evaluate_tests
    evaluate_tests(test_results, assert_test=False)
    # Passed chi2 test.
    # Passed exist test.

    changed_model_output = choice([0, 1, 2], size=(1000,))
    changed_model_output[:200] = 0
    changed_test_results = ce.check_data(changed_model_output)
    # Failed chi2 check; test statistic=59.0655, p=0.0000
    # Passed min check; observed=[0 1 2] (Expected [0, 1, 2])

    evaluate_tests(changed_test_results, assert_test=False)
    # Failed chi2 test.
    # Passed exist test.

Updating test parameters
========

Example of changing the minimum expected value to 0. I demonstrate the three different ways this can be done.

.. code-block:: python3

    from predeval import ContinuousEvaluator
    ce = ContinuousEvaluator(model_output)
    ce.update_param('minimum', 0)

    # or

    ce.assertion_params['minimum'] = 0

    # or

    ce.update_min([0])

Example of changing the maximum expected value to 100.

.. code-block:: python3

    from predeval import ContinuousEvaluator
    ce = ContinuousEvaluator(model_output)
    ce.update_param('maximum', 100)

Example of changing the expected mean to 50.

.. code-block:: python3

    from predeval import ContinuousEvaluator
    ce = ContinuousEvaluator(model_output)
    ce.update_param('mean', 50)

Example of changing expected standard-deviation to 10.

.. code-block:: python3

    from predeval import ContinuousEvaluator
    ce = ContinuousEvaluator(model_output)
    ce.update_param('std', 10)

Example of changing Kolmogorov-Smirnov test threshold to 1.

.. code-block:: python3

    from predeval import ContinuousEvaluator
    ce = ContinuousEvaluator(model_output)
    ce.update_param('ks_stat', 1)

Example of changing Kolmogorov-Smirnov test.

.. code-block:: python3

    from predeval import ContinuousEvaluator
    ce = ContinuousEvaluator(model_output)
    ce.update_ks_test(new_model_output)

Example of changing Chi-square test of independence threshold to 3.

.. code-block:: python3

    from predeval import CategoricalEvaluator
    ce = CategoricalEvaluator(model_output)
    ce.update_chi2_test(new_model_output)

Example of changing Chi-square test.

.. code-block:: python3

    from predeval import CategoricalEvaluator
    ce = CategoricalEvaluator(model_output)
    ce.update_param('chi2_stat', 3)

Example of changing expected categories to 1, 2, and 3.

.. code-block:: python3

    from predeval import CategoricalEvaluator
    ce = CategoricalEvaluator(model_output)
    ce.update_param('cat_exists', [1, 2, 3])


Changing evaluation tests
========

You might not want to run the entire test suite. Here's some examples of how to change what tests are run.

.. code-block:: python3

    from predeval import ContinuousEvaluator
    ce = ContinuousEvaluator(model_output, assertions=['min', 'max'])

    # or you can run the tests one at a time.

    ce.check_min(new_model_output)
    ce.check_max(new_model_output)

Saving and Loading your evaluator
========

Here's an example of how to save and load your evaluator.

.. code-block:: python3

    from predeval import ContinuousEvaluator
    ce = ContinuousEvaluator(model_output)

    from joblib import dump, load
    dump(ce, 'con_eval.joblib')  # save evaluator
    ce = load('con_eval.joblib')  # load evaluator
