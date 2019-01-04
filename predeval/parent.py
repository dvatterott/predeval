"""Library of classes for evaluating continuous model outputs."""
from abc import ABCMeta, abstractproperty
import numpy as np

__author__ = 'Dan Vatterott'
__license__ = 'MIT'


class ParentPredEval(object):
    """
    Parent Class for evaluator classes.
    """
    __metaclass__ = ABCMeta

    @abstractproperty
    def _possible_assertions(self):
        raise NotImplementedError

    @abstractproperty
    def assertions(self):  # pylint: disable=C0111
        raise NotImplementedError

    @abstractproperty
    def assertion_params(self):  # pylint: disable=C0111
        raise NotImplementedError

    @abstractproperty
    def _tests(self):
        raise NotImplementedError

    def __init__(
            self,
            ref_data,
            verbose=True):
        assert isinstance(verbose, bool), 'expected boolean, input verbose is not a boolean'
        self.verbose = verbose

        assert len(ref_data.shape) == 1, 'Input data not a single vector'
        self.ref_data = np.array(ref_data)

    def _check_assertion_types(self, assertions):
        """Check whether test_data is as expected.

        Parameters
        ----------
        assertions : list or string
            These are the assertions we want to test.

        Returns
        -------
        list of strings describing assertions to evaluate.

        """
        assert isinstance(assertions, (str, list)), 'assertions given in unexpected type'
        assertions = list(assertions) if isinstance(assertions, str) else assertions
        assert all([x in self._possible_assertions
                    for x in assertions]), 'unexpected assertion request'
        return assertions

    def check_data(self, test_data):
        """Check whether test_data is as expected.

        Run threw all tests in assertions and return whether the data passed these tests.

        Parameters
        ----------
        test_data : list or np.array
            This the data that will be compared to the reference data.

        Returns
        -------
        output : list of tuples
            Each tuple has a string a boolean. The string describes the test.
            The boolean describes the outcome. True is a pass and False is a fail.

        """
        test_data = np.array(test_data) if isinstance(test_data, list) else test_data
        assert len(test_data.shape) == 1, 'Input data not a single vector'
        output = []
        for funs in self._tests:
            output.append(funs(comparison_data=test_data))
        return output

    def update_param(self, param_key, param_value):
        """Update value in assertion param dictionary attribute.

        Parameters
        ----------
        param_key : string
            This is the assertion param that we want to update.
        param_value : real number or partially evaluated test.
            This is the updated value.

        Returns
        -------
        None

        """
        assert param_key in self.assertion_params, 'Requested key is not in assertion_params dict'
        self.assertion_params[param_key] = param_value
