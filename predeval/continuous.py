"""Library of classes for evaluating continuous model outputs."""
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


class ContinuousEvaluator(ParentPredEval):
    """
    Evaluator for continuous model outputs (e.g., regression models).
    """
    def __init__(
            self,
            ref_data,
            assertions=None,
            verbose=True,
            **kwargs):
        super(ContinuousEvaluator, self).__init__(ref_data, verbose=verbose)

        # ---- Fill in Assertion Parameters ---- #
        self.assertion_params_ = {
            'minimum': kwargs.get('min', None),
            'maximum': kwargs.get('max', None),
            'mean': kwargs.get('mean', None),
            'std': kwargs.get('std', None),
            'ks_test': None
        }

        assert isinstance(kwargs.get('ks_stat', 0.2),
                          Real), 'expected number, input ks_test_stat is not a number'
        self.assertion_params_['ks_stat'] = kwargs.get('ks_stat', 0.2)

        # ---- create list of assertions to test ---- #
        self.possible_assertions_ = {
            'min': (self.find_min, self.check_min),
            'max': (self.find_max, self.check_max),
            'mean': (self.find_mean, self.check_mean),
            'std': (self.find_std, self.check_std),
            'ks_test': (self.create_ks_test, self.check_ks),
        }

        # ---- create list of assertions to test ---- #
        assertions = ['min', 'max', 'mean', 'std', 'ks_test'] if assertions is None else assertions
        self.assertions_ = self.check_assertion_types(assertions)

        # ---- populate assertion tests with reference data ---- #
        for i in self.assertions_:
            self.possible_assertions[i][0](self.ref_data)

        if ('std' not in assertions) and ('mean' in assertions):
            self.possible_assertions['std'][0](self.ref_data)

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

    def create_ks_test(self, input_data):
        """Create partially evaluated ks_test.

        Parameters
        ----------
        input_data : list (ideally one-dimensional np.array)
            This the reference data for the ks-test. All future data will be compared to this data.

        Returns
        -------
        None

        """
        assert len(input_data) >= 25, 'Not enough data for reliable KS tests'
        self.assertion_params['ks_test'] = partial(stats.ks_2samp, np.array(input_data))

    def find_min(self, input_data):
        """Find min of input_data.

        Parameters
        ----------
        input_data : list (ideally one-dimensional np.array)
            This the reference data for the min-test. All future data will be compared to this data.

        Returns
        -------
        None

        """

        if self.assertion_params['minimum'] is None:
            assert len(input_data.shape) == 1, 'Input data not a single vector'
            self.assertion_params['minimum'] = np.min(input_data)

    def find_max(self, input_data):
        """Find max of input data.

        Parameters
        ----------
        input_data : list (ideally one-dimensional np.array)
            This the reference data for the max-test. All future data will be compared to this data.

        Returns
        -------
        None

        """
        if self.assertion_params['maximum'] is None:
            assert len(input_data.shape) == 1, 'Input data not a single vector'
            self.assertion_params['maximum'] = np.max(input_data)

    def find_mean(self, input_data):
        """Find mean of input data.

        Parameters
        ----------
        input_data : list (ideally one-dimensional np.array)
            This the reference data for the max-test. All future data will be compared to this data.

        Returns
        -------
        None

        """
        if self.assertion_params['mean'] is None:
            assert len(input_data.shape) == 1, 'Input data not a single vector'
            self.assertion_params['mean'] = np.mean(input_data)

    def find_std(self, input_data):
        """Find standard deviation of input data.

        Parameters
        ----------
        input_data : list (ideally one-dimensional np.array)
            This the reference data for the max-test. All future data will be compared to this data.

        Returns
        -------
        None

        """
        if self.assertion_params['std'] is None:
            assert len(input_data.shape) == 1, 'Input data not a single vector'
            self.assertion_params['std'] = np.std(input_data)

    def check_min(self, comparison_data=None):
        """Check whether test_data has any smaller values than expected.

        Parameters
        ----------
        comparison_data : list (ideally one-dimensional np.array)
            This the data that will be compared to the reference data.

        Returns
        -------
        (string, bool)
            2 item tuple with test name and boolean expressing whether passed test.

        """
        assert self.assertion_params['minimum'] is not None, 'Must input or load reference minimum'
        test_data = self.ref_data if comparison_data is None else comparison_data
        min_obs = np.min(np.array(test_data))
        passed = True if min_obs >= self.assertion_params['minimum'] else False
        pass_fail = 'Passed' if passed else 'Failed'
        if self.verbose:
            print('{} min check; min observed={}'.format(pass_fail, min_obs))
        return ('min', passed)

    def check_max(self, comparison_data=None):
        """Check whether test_data has any larger values than expected.

        Parameters
        ----------
        comparison_data : list (ideally one-dimensional np.array)
            This the data that will be compared to the reference data.

        Returns
        -------
        (string, bool)
            2 item tuple with test name and boolean expressing whether passed test.

        """
        assert self.assertion_params['maximum'] is not None, 'Must input or load reference maximum'
        test_data = self.ref_data if comparison_data is None else comparison_data
        max_obs = np.max(np.array(test_data))
        passed = True if max_obs <= self.assertion_params['maximum'] else False
        pass_fail = 'Passed' if passed else 'Failed'
        if self.verbose:
            print('{} max check; max observed={}'.format(pass_fail, max_obs))
        return ('max', passed)

    def check_mean(self, comparison_data=None):
        """Check whether test_data has a different mean than expected.

        Parameters
        ----------
        comparison_data : list (ideally one-dimensional np.array)
            This the data that will be compared to the reference data.

        Returns
        -------
        (string, bool)
            2 item tuple with test name and boolean expressing whether passed test.

        """
        assert self.assertion_params['mean'] is not None, 'Must input or load reference mean'
        assert self.assertion_params['std'] is not None, 'Must input or load reference mean'
        test_data = self.ref_data if comparison_data is None else comparison_data
        mean_obs = np.mean(np.array(test_data))

        two_std = self.assertion_params['std'] * 2

        passed = [False, False]
        passed[0] = True if mean_obs >= self.assertion_params['mean'] - two_std else False
        passed[1] = True if mean_obs <= self.assertion_params['mean'] + two_std else False

        pass_fail = 'Passed' if all(passed) else 'Failed'
        if self.verbose:
            print('{} mean check; mean observed={} (Expected {} +- {})'.format(
                pass_fail,
                mean_obs,
                self.assertion_params['mean'],
                two_std))
        return ('mean', all(passed))

    def check_std(self, comparison_data=None):
        """Check whether test_data has any larger values than expected.

        Parameters
        ----------
        comparison_data : list (ideally one-dimensional np.array)
            This the data that will be compared to the reference data.

        Returns
        -------
        (string, bool)
            2 item tuple with test name and boolean expressing whether passed test.

        """
        assert self.assertion_params['std'] is not None, 'Must input or load reference std'
        test_data = self.ref_data if comparison_data is None else comparison_data
        std_obs = np.std(np.array(test_data))

        half_std = self.assertion_params['std'] * 0.5

        passed = [False, False]
        passed[0] = True if std_obs >= self.assertion_params['std'] - half_std else False
        passed[1] = True if std_obs <= self.assertion_params['std'] + half_std else False

        pass_fail = 'Passed' if all(passed) else 'Failed'
        if self.verbose:
            print('{} std check; std observed={} (Expected {} +- {})'.format(
                pass_fail,
                std_obs,
                self.assertion_params['std'],
                half_std))
        return ('std', all(passed))

    def check_ks(self, comparison_data=None):
        """Test whether test_data is similar to reference data.

        Parameters
        ----------
        comparison_data : list (ideally one-dimensional np.array)
            This the data that will be compared to the reference data.

        Returns
        -------
        (string, bool)
            2 item tuple with test name and boolean expressing whether passed test.

        """
        assert self.assertion_params['ks_test'], 'Must input or load reference data ks-test'
        test_data = self.ref_data if comparison_data is None else comparison_data
        assert len(test_data) >= 25, 'Not enough data for reliable KS tests'
        test_stat, p_value = self.assertion_params['ks_test'](np.array(test_data))  # pylint: disable=E1102
        passed = True if test_stat <= self.assertion_params['ks_stat'] else False
        pass_fail = 'Passed' if passed else 'Failed'
        if self.verbose:
            print('{} ks check; test statistic={}, p={}'.format(pass_fail, test_stat, p_value))
        return ('ks', passed)
