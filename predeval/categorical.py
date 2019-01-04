"""Library of classes for evaluating categorical model outputs."""
from numbers import Real
from functools import partial
import numpy as np
from scipy import stats
from .parent import ParentPredEval

"""
https://scikit-learn.org/stable/modules/model_persistence.html
from joblib import dump, load
dump(clf, 'filename.joblib')
clf = load('filename.joblib')
"""

__author__ = 'Dan Vatterott'
__license__ = 'MIT'


def _chi2_test(reference, test_data):
    """Change chi2_contingency inputs for partial evaluation.

    Parameters
    ----------
    reference : list or np.array
        This the reference data that will be used for the comparison.
    test_data : list or np.array
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
    return stats.chi2_contingency(obs)


class CategoricalEvaluator(ParentPredEval):
    """
    Evaluator for categorical model outputs (e.g., classification models).

    ...

    Parameters
    ----------
    ref_data : list of int or float or np.array
        This the reference data for all tests. All future data will be compared to this data.
    assertions : list of str, optional
        These are the assertion tests that will be created. Defaults is ['chi2_test', 'exist'].
    verbose : bool, optional
        Whether tests should print their output. Default is true

    Attributes
    ----------
    assertion_params : dict
        dictionary of test names and values defining these tests
        (e.g., test-statistic for chi2_test).
        Default value for chi2_test is 0.2.
    assertions : list of str
        This list of strings describes the tests that will be run on comparison data.

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
        self._possible_assertions_ = {
            'chi2_test': (self.update_chi2_test, self.check_chi2),
            'exist': (self.update_exist, self.check_exist),
        }

        # ---- create list of assertions to test ---- #
        assertions = ['chi2_test', 'exist'] if assertions is None else assertions
        self.assertions_ = self._check_assertion_types(assertions)

        # ---- populate assertion tests with reference data ---- #
        for i in self.assertions_:
            self._possible_assertions[i][0](self.ref_data)

        # ---- populate list of tests to run and run tests ---- #
        self._tests_ = [self._possible_assertions_[i][1] for i in self.assertions_]

    @property
    def assertion_params(self):
        return self.assertion_params_

    @property
    def _possible_assertions(self):
        return self._possible_assertions_

    @property
    def assertions(self):
        return self.assertions_

    @property
    def _tests(self):
        return self._tests_

    def update_chi2_test(self, input_data):
        """Create partially evaluated chi2 contingency test.

        Parameters
        ----------
        input_data : list or np.array
            This the reference data for the ks-test. All future data will be compared to this data.

        Returns
        -------
        None

        """
        assert len(input_data.shape) == 1, 'Input data not a single vector'
        _, counts = np.unique(input_data, return_counts=True)
        assert all([x >= 5 for x in counts]), \
            'Not enough data of each type for reliable Chi2 Contingency test. Need at least 5.'
        self.assertion_params['chi2_test'] = partial(_chi2_test, np.array(counts))

    def update_exist(self, input_data):
        """Create input data for test checking whether all categorical outputs exist.

        Parameters
        ----------
        input_data : list or np.array
            This the reference data for the check_exist. All future data will be compared to it.

        Returns
        -------
        None

        """
        if self.assertion_params['cat_exists'] is None:
            assert len(input_data.shape) == 1, 'Input data not a single vector'
            self.assertion_params['cat_exists'] = np.unique(input_data)

    def check_chi2(self, test_data):
        """Test whether test_data is similar to reference data.

        If chi2-test-statistic exceeds the value in assertion_params,
        then the test will produce a False (rather than True).

        Parameters
        ----------
        test_data : list or np.array
            This the data that will be compared to the reference data.

        Returns
        -------
        (string, bool)
            2 item tuple with test name and boolean expressing whether passed test.

        """
        assert self.assertion_params['chi2_test'], 'Must input or load reference data chi2-test'
        assert len(test_data.shape) == 1, 'Input data not a single vector'
        _, counts = np.unique(test_data, return_counts=True)
        assert all([x >= 5 for x in counts]), \
            'Not enough data of each type for reliable Chi2 Contingency test. '\
            'Need at least 5 values in each cell.'
        test_stat, p_value, _, _ = self.assertion_params['chi2_test'](counts)  # pylint: disable=E1102
        passed = True if test_stat <= self.assertion_params['chi2_stat'] else False
        pass_fail = 'Passed' if passed else 'Failed'
        if self.verbose:
            print('{} chi2 check; test statistic={}, p={}'.format(pass_fail, test_stat, p_value))
        return ('chi2', passed)

    def check_exist(self, test_data):
        """Check that all distinct values present in test_data.

        If any values missing, then the function will return a False (rather than true).

        Parameters
        ----------
        test_data : list or np.array
            This the data that will be compared to the reference data.

        Returns
        -------
        (string, bool)
            2 item tuple with test name and boolean expressing whether passed test.
        """
        assert self.assertion_params['cat_exists'] is not None,\
            'Must input or load reference minimum'
        assert len(test_data.shape) == 1, 'Input data not a single vector'
        obs = np.unique(np.array(test_data))
        exp = list(self.assertion_params['cat_exists'])
        passed = True if all([x in exp for x in obs]) and all([x in obs for x in exp]) else False
        pass_fail = 'Passed' if passed else 'Failed'
        if self.verbose:
            print('{} min check; observed={} (Expected {})'.format(pass_fail, obs, exp))
        return ('exist', passed)
