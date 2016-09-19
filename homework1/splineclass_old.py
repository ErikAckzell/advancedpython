# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 07:20:29 2016

@author: Erik Ackzell
"""

import pylab
import numpy
import scipy


class spline(object):
    def __init__(self, controlpoints, grid):
        self.controlpoints = controlpoints
        self.grid = grid

    def __call__(self):
        pass

    def plot(self):
        pass

    def get_alpha(self, u, indices):
        alpha = ((self.grid[indices[0]] - u) /
                 (self.grid[indices[0]] - self.grid[indices[1]]))
        return alpha

    def evaluate(self, u):
        index = self.get_index(u)
        current_controlpoints = self.get_current_controlpoints(u)
        value_matrix = scipy.column_stack((current_controlpoints,
                                           scipy.zeros((4, 3))))
        for i in range(1, 4):
            for j in range(i):
                leftmostknot = index + i - 3
                rightmostknot = leftmostknot + 4 - j
                alpha = self.get_alpha(u, [leftmostknot, rightmostknot])
                value_matrix[i, j] = (
                                      alpha * value_matrix[i-1, j-1] +
                                      (1 - alpha) * value_matrix[i, j-1]
                                     )

    def get_basisfunction(self, knots_in, index_in, degree_in):
        def basisfunction(u,
                          knots=numpy.copy(knots_in),
                          index=index_in,
                          degree=degree_in):
            if degree == 0:
                if knots[index - 1] <= u < knots[index]:
                    return 1
                else:
                    return 0
            else:
                return (u - knots[index - 1]) \
                    * basisfunction(u, knots, index, degree - 1) \
                    / (knots[index + degree - 1] - knots[index - 1]) \
                    + (knots[index + degree] - u) \
                    * basisfunction(u, knots, index + 1, degree - 1) \
                    / (knots[index + degree] - knots[index])

        return basisfunction

    def get_current_control_points(self, u):
        return numpy.arange(0, 4)

#s = spline(2)
#knots_in = list(range(10))
#degree_in = 3
#
#func1 = s.get_basisfunction(knots_in=knots_in,
#                            index_in=3,
#                            degree_in=degree_in)
#
#func2 = s.get_basisfunction(knots_in=knots_in,
#                            index_in=4,
#                            degree_in=degree_in)
#
#
#ulist1 = scipy.linspace(0, 10, 100)
#ylist1 = [func1(u) for u in ulist1]
#
#pylab.plot(ulist1, ylist1)
#
#
#ulist2 = scipy.linspace(0, 10, 100)
#ylist2 = [func2(u) for u in ulist2]
#
#pylab.plot(ulist2, ylist2)
