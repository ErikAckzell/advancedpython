# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 10:18:11 2016

@author: Erik Ackzell
"""

import scipy
import unittest


def rosenbrockfunction(x):
    """
    This is the Rosenbrock function R^2 -> R.
    Input x is an iterable of lenght 2
    """
    x = scipy.array(x)
    try:
        x = x.reshape((2, 1))
    except ValueError:
        raise ValueError('Input has wrong number of elements')
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def rosenbrockderivative(x):
    pass

class TestRosenbrock(unittest.TestCase):
    def test_wrong_input(self):
        x = [0, 0]
        while len(scipy.array(x).flatten()) == 2:
            dim1 = scipy.random.randint(0, 2000)
            dim2 = scipy.random.randint(0, 2000)
            x = scipy.random.random((dim1, dim2))
        print(x)
        self.assertRaises(ValueError,
                          rosenbrockfunction, x)

    def test_returnval(self):
        x = scipy.random.random((2, 1))
        result = 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
        self.assertEqual(rosenbrockfunction(x),
                         result)

if __name__ == '__main__':
    unittest.main()
