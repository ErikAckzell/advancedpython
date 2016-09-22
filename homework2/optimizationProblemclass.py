# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 12:48:02 2016

@author: Erik Ackzell
"""


class optimizationProblem(object):
    """
    This is a class for optimization problems.
    """
    def __init__(self, function, gradient=None):
        """
        An object of the class is initialized with an objective function and
        optionally the gradient of the function.
        """
        self.f = function
        self.g = gradient
