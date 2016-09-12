# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 15:33:33 2016

@author: Erik Ackzell
"""

import scipy
import pylab


class Bspline(object):
    def __init__(self, grid, controlpoints):
        """
        grid (iterable): grid points should have multiplicity 3 in order to
        have the spline starting and ending in the first and last control
        point, respectively.
        controlpoints (array): should be on the form
                controlpoints = array([
                                      [d_00, d01],
                                      [d_10, d11],
                                      ...
                                      [d_L0, d_L1],
                                      ]),
        i.e. an (L+1)x2 array.
        """
        if len(grid) != len(controlpoints) + 2:
            raise ValueError('Number of gridpoints or controlpoints is wrong')
        self.grid = grid
        self.controlpoints = controlpoints

    def __call__(self, u):
        """
        Method to evaluate the spline at point u, using de Boor's algorithm.
        """
        index = self.get_index(u)
        current_controlpoints = self.get_controlpoints(index)
        value_matrix = scipy.column_stack((current_controlpoints,
                                           scipy.zeros((4, 6))))
        for i in range(1, 4):
            for j in range(1, i + 1):
                leftmostknot = index + i - 3
                rightmostknot = leftmostknot + 4 - j
                alpha = self.get_alpha(u, [leftmostknot, rightmostknot])
                value_matrix[i, j*2:j*2+2] = (
                            alpha * value_matrix[i-1, (j-1)*2:(j-1)*2+2] +
                            (1 - alpha) * value_matrix[i, (j-1)*2:(j-1)*2+2]
                                             )
        return value_matrix[3, -2:]

    def get_controlpoints(self, index):
        """
        Method to obtain the current control points for de Boor's algorithm.
        index (int): the index depending on the point u at which to evaluate
        the spline (see get_index method).
        """
        if index < 2:
            current_controlpoints = self.controlpoints[0:4]
        elif index > len(self.controlpoints) - 2:
            current_controlpoints = self.controlpoints[-4:]
        else:
            current_controlpoints = self.controlpoints[index - 2:index + 2]
        return current_controlpoints

    def get_alpha(self, u, indices):
        """
        Returns the alpha parameter used for linear interpolation of the
        values in the de Boor scheme.
        u (float): value at which to evaluate the spline
        indices (iterable): indices for the leftmost and rightmost knots
        corresponding to the current blossom pair
        """
        indices[0] = max(0, indices[0])
        indices[1] = min(len(self.grid) - 1, indices[1])
        try:
            alpha = ((self.grid[indices[1]] - u) /
                     (self.grid[indices[1]] - self.grid[indices[0]]))
        except ZeroDivisionError:
            alpha = 0
        return alpha

    def get_index(self, u):
        """
        Method to get the index of the grid point at the left endpoint of the
        gridpoint interval at which the current value u is. If u belongs to
            [u_I, u_{I+1}]
        it returns the index I.
        u (float): value at which to evaluate the spline
        """
        if u == self.grid[-1]:
            index = len(self.grid) - 2
        else:
            index = (self.grid > u).argmax() - 1
        return index

    def plot(self, points=300, controlpoints=True):
        """
        Method to plot the spline.
        points (int): number of points to use when plotting the spline
        controlpoints (bool): if True, plots the controlpoints as well
        """
        ulist = scipy.linspace(self.grid[0], self.grid[-1], points)
        pylab.plot(*zip(*[self(u) for u in ulist]))
        if controlpoints:
            pylab.plot(*zip(*self.controlpoints), 'o--')


controlpoints = scipy.array([[40, 17],
                             [20, 0],
                             [18, 8],
                             [57, -27],
                             [8, -77],
                             [-23, -65],
                             [-100, -15],
                             [-23, 7],
                             [-40, 20],
                             [-15, 10]])
grid = scipy.hstack((scipy.zeros(2),
                     scipy.linspace(0, 1, len(controlpoints) + 2 - 4),
                     scipy.ones(2)))
spline = Bspline(grid=grid, controlpoints=controlpoints)
spline.plot()
