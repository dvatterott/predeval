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
    def possible_assertions(self):
        """Make sure possible_assertions property is implemented."""
        raise NotImplementedError

    @abstractproperty
    def assertions(self):
        """Make sure assertions property is implemented."""
        raise NotImplementedError

    @abstractproperty
    def assertion_params(self):
        """Make sure assertion_params property is implemented."""
        raise NotImplementedError

    @abstractproperty
    def tests(self):
        """Make sure tests property is implemented."""
        raise NotImplementedError

    def __init__(
            self,
            ref_data,
            verbose=True):
        assert isinstance(verbose, bool), 'expected boolean, input verbose is not a boolean'
        self.verbose = verbose

        assert len(ref_data.shape) == 1, 'Input data not a single vector'
        self.ref_data = np.array(ref_data)

    def check_assertion_types(self, assertions):
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
        assert all([x in self.possible_assertions
                    for x in assertions]), 'unexpected assertion request'
        return assertions

    def check_data(self, comparison_data=None):
        """Check whether test_data is as expected.

        Parameters
        ----------
        comparison_data : list (ideally one-dimensional np.array)
            This the data that will be compared to the reference data.

        Returns
        -------
        None

        """
        test_data = self.ref_data if comparison_data is None else comparison_data
        assert len(test_data.shape) == 1, 'Input data not a single vector'
        output = []
        for funs in self.tests:
            output.append(funs(comparison_data=test_data))
        return output
