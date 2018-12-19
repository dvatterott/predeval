"""Library of classes for evaluating categorical model outputs."""
from numbers import Real
from functools import partial
import numpy as np
import scipy
from .parent import ParentPredEval

"""
https://scikit-learn.org/stable/modules/model_persistence.html
from joblib import dump, load
dump(clf, 'filename.joblib')
clf = load('filename.joblib')
"""

__author__ = 'Dan Vatterott'
__license__ = 'MIT'


def chi2_test(reference, test_data):
    """Change chi2_contingency inputs for partial evaluation.

    Parameters
    ----------
    reference : list (ideally one-dimensional np.array)
        This the reference data that will be used for the comparison.
    test_data : list (ideally one-dimensional np.array)
        This the data compared to the reference data.

    Returns
    -------
    chi2 : float
        The test statistic.
    p : float
        The p-value of the test
    dof : int
        Degrees of freedom
    expected : ndarray, same shape as `observed`
        The expected frequencies, based on the marginal sums of the table.

    """
    obs = np.append([reference], [test_data], axis=0)
    return scipy.stats.chi2_contingency(obs)


class CategoricalEvaluator(ParentPredEval):
    """
    Evaluator for categorical model outputs (e.g., classification models).
    """
    def __init__(
            self,
            ref_data,
            assertions=None,
            verbose=True,
            **kwargs):
        super(CategoricalEvaluator, self).__init__(ref_data, verbose=verbose)

        # ---- Fill in Assertion Parameters ---- #
        self.assertion_params_ = {
            'chi2_test': None,
            'cat_exists': None,
        }

        assert isinstance(kwargs.get('chi2_stat', 0.2),
                          Real), 'expected number, input chi2_test_stat is not a number'
        self.assertion_params_['chi2_stat'] = kwargs.get('chi2_stat', 0.2)

        # ---- create list of assertions to test ---- #
        self.possible_assertions_ = {
            'chi2_test': (self.create_chi2_test, self.check_chi2),
            'cat_exists': (self.create_exist, self.check_exist),
        }

        # ---- create list of assertions to test ---- #
        assertions = ['chi2_test', 'exist'] if assertions is None else assertions
        self.assertions_ = self.check_assertion_types(assertions)

        # ---- populate assertion tests with reference data ---- #
        for i in self.assertions_:
            self.possible_assertions[i][0](self.ref_data)

        # ---- populate list of tests to run and run tests ---- #
        self.tests_ = [self.possible_assertions_[i][1] for i in self.assertions_]

    @property
    def assertion_params(self):
        return self.assertion_params_

    @property
    def possible_assertions(self):
        return self.possible_assertions_

    @property
    def assertions(self):
        return self.assertions_

    @property
    def tests(self):
        return self.tests_

    def create_chi2_test(self, input_data):
        """Create partially evaluated chi2 contingency test.

        Parameters
        ----------
        input_data : list (ideally one-dimensional np.array)
            This the reference data for the ks-test. All future data will be compared to this data.

        Returns
        -------
        None

        """
        assert len(input_data.shape) == 1, 'Input data not a single vector'
        _, counts = np.unique(input_data, return_counts=True)
        assert all([x >= 5 for x in counts]), \
            'Not enough data of each type for reliable Chi2 Contingency test. Need at least 5.'
        self.assertion_params['chi2_test'] = partial(chi2_test, np.array(input_data))

    def create_exist(self, input_data):
        """Create input data for test checking whether all categorical outputs exist.

        Parameters
        ----------
        input_data : list (ideally one-dimensional np.array)
            This the reference data for the check_exist. All future data will be compared to it.

        Returns
        -------
        None

        """
        if self.assertion_params['cat_exists'] is None:
            assert len(input_data.shape) == 1, 'Input data not a single vector'
            self.assertion_params['cat_exists'] = np.unique(input_data)

    def check_chi2(self, comparison_data=None):
        """Test whether test_data is similar to reference data.

        Parameters
        ----------
        comparison_data : list (ideally one-dimensional np.array)
            This the data that will be compared to the reference data.

        Returns
        -------
        None

        """
        assert self.assertion_params['chi2_test'], 'Must input or load reference data ks-test'
        test_data = self.ref_data if comparison_data is None else comparison_data
        assert len(test_data.shape) == 1, 'Input data not a single vector'
        _, counts = np.unique(test_data, return_counts=True)
        test_stat, p_value = self.assertion_params['chi2_test'](counts)  # pylint: disable=E1102
        assert test_stat <= self.assertion_params['chi2_stat'],\
            'Distribution different than expected'
        if self.verbose:
            print('Passed ks check; test statistic={}, p={}'.format(test_stat, p_value))

    def check_exist(self, comparison_data=None):
        """Check that all distinct values present in test_data.

        Parameters
        ----------
        comparison_data : list (ideally one-dimensional np.array)
            This the data that will be compared to the reference data.

        Returns
        -------
        2 item tuple with test name and boolean expressing whether passed test.

        """
        assert self.assertion_params['cat_exists'] is not None,\
            'Must input or load reference minimum'
        test_data = self.ref_data if comparison_data is None else comparison_data
        obs = np.unique(np.array(test_data))
        exp = list(self.assertion_params['cat_exists'])
        passed = True if all([x in exp for x in obs]) and all([x in obs for x in exp]) else False
        pass_fail = 'Passed' if passed else 'Failed'
        if self.verbose:
            print('{} min check; observed={} (Expected {})'.format(pass_fail, obs, exp))
        return ('exist', passed)
