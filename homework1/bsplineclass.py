# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 15:33:33 2016

@author: Erik Ackzell
"""

import scipy
import pylab
import unittest
import numpy as np


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
        try:
            grid = scipy.array(grid)
            grid = grid.reshape((len(grid), 1))
        except ValueError:
            raise ValueError('Grid should be a one-dimensional list or array')
        if len(grid) != len(controlpoints) + 2:
            raise ValueError('Number of gridpoints or controlpoints is wrong')
        if controlpoints.shape[1] != 2:
            raise ValueError('Controlpoints should be an (L+1)x2 array.')
        self.grid = grid.reshape((len(grid), 1))
        self.controlpoints = controlpoints

    def __call__(self, u):
        """
        Method to evaluate the spline at point u, using de Boor's algorithm.
        """
        # get index of grid point left of u
        index = self.get_index(u)
        # get current controlpoints
        current_controlpoints = self.get_controlpoints(index)
        # setup matrix to store the values in the de Boor array:
        # deBoorvalues =
        #              d[I-2, I-1, I]
        #              d[I-1, I, I+1]   d[u, I-1, I]
        #              d[I, I+1, I+2]   d[u, I, I+1]   d[u, u, I]
        #              d[I+1, I+2, I+3] d[u, I+1, I+2] d[u, u, I+1] d[u, u, u]
        deBoorvalues = scipy.column_stack((current_controlpoints,
                                           scipy.zeros((4, 6))))
        # calculate values for de Boor array
        for i in range(1, 4):
            for j in range(1, i + 1):
                leftmostknot = index + i - 3  # current leftmost knot
                rightmostknot = leftmostknot + 4 - j  # current rightmost knot
                alpha = self.get_alpha(u, [leftmostknot, rightmostknot])
                deBoorvalues[i, j*2:j*2+2] = (
                            alpha * deBoorvalues[i-1, (j-1)*2:(j-1)*2+2] +
                            (1 - alpha) * deBoorvalues[i, (j-1)*2:(j-1)*2+2]
                                             )
        return deBoorvalues[3, -2:]

    def get_basisfunc(self, j, knots, u, k=3):
        """
        Method to evaluate the the basis function N^k with index j at point u.
        j (int): the index of the basis function we want to evaluate
        knots (array): knot sequence u_i, where i=0,...,K
        u (float): the point where to evaluate the basis function
        k (int): the degree of the basis function
        """
        if k == 0:
            return 1 if knots[j] <= u < knots[j+1] \
                     else 0
        else:
            try:
                basisfunc = (u - knots[j])/(knots[j+k]-knots[j]) * self.get_basisfunc(j,knots,u,k=k-1) \
                + (knots[j+k+1] - u)/(knots[j+k+1] - knots[j+1]) * self.get_basisfunc(j+1,knots,u,k=k-1)
            except ZeroDivisionError:
                basisfunc = 0
            except IndexError:
                numBasisfunc = len(knots) - 1 - k
                basisfunc = 'Invalid index. There are no more than {} basis functions for the given problem, choose an ' \
                            'index lower than the number of basis functions.'.format(numBasisfunc)
            return basisfunc

    def get_controlpoints(self, index):
        """
        Method to obtain the current control points for de Boor's algorithm.
        index (int): the index depending on the point u at which to evaluate
        the spline (see get_index method).
        """
        if index < 2:  # is index in very beginning
            current_controlpoints = self.controlpoints[0:4]  # use first points
        elif index > len(self.controlpoints) - 2:  # is index in very end
            current_controlpoints = self.controlpoints[-4:]  # use last points
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
        indices[0] = max(0, indices[0])  # adjust for very beginning
#        indices[1] = min(len(self.grid) - 1, indices[1])  # adjust for very end
        try:
            alpha = ((self.grid[indices[1]] - u) /
                     (self.grid[indices[1]] - self.grid[indices[0]]))
        except ZeroDivisionError:  # catch multiplicity of knots
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
        if u == self.grid[-1]:  # check if u equals last knot
#            index = len(self.grid) - 2  # pick next to last index
            index = (self.grid < u).argmin() - 1
        else:
            index = (self.grid > u).argmax() - 1
        return index

    def plot(self, points=300, controlpoints=True):
        """
        Method to plot the spline.
        points (int): number of points to use when plotting the spline
        controlpoints (bool): if True, plots the controlpoints as well
        """
        # list of u values for which to plot
        ulist = scipy.linspace(self.grid[0], self.grid[-1], points)
        pylab.plot(*zip(*[self(u) for u in ulist]))
        if controlpoints:  # checking whether to plot control points
            pylab.plot(*zip(*self.controlpoints), 'o--')
            pylab.show()


class TestBsplineClass(unittest.TestCase):
    def get_random_controlpoints(self, second_dimension):
        controlpoints = scipy.random.random((scipy.random.randint(10, 200),
                                             second_dimension))
        return controlpoints

    def get_random_grid(self, length):
        a = scipy.random.randint(-1000, 1000) * scipy.random.random()
        b = a + abs(scipy.random.randint(10) * scipy.random.random())
        grid = scipy.linspace(a, b, length)
        return grid

    def get_spline(self):
        controlpoints = self.get_random_controlpoints(2)
        grid = self.get_random_grid(len(controlpoints) + 2)
        return Bspline(grid=grid, controlpoints=controlpoints)

    def test_init(self):
        self.get_spline()

    def test_wrong_grid1(self):
        controlpoints = self.get_random_controlpoints(2)
        grid = scipy.random.random((3, 3))
        self.assertRaises(ValueError,
                          Bspline, grid=grid, controlpoints=controlpoints)

    def test_wrong_grid2(self):
        controlpoints = self.get_random_controlpoints(2)
        grid = self.get_random_grid(len(controlpoints) - 1)
        self.assertRaises(ValueError,
                          Bspline, grid=grid, controlpoints=controlpoints)

    def test_wrong_controlpoints(self):
        controlpoints = self.get_random_controlpoints(3)
        grid = scipy.linspace(0, 10, len(controlpoints) + 2)
        self.assertRaises(ValueError,
                          Bspline, grid=grid, controlpoints=controlpoints)

    def test_get_index1(self):
        controlpoints = scipy.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        grid = scipy.array([1, 2, 3, 4, 5, 6, 7])
        spline = Bspline(grid=grid, controlpoints=controlpoints)
        self.assertEqual(spline.get_index(1.5), 0)

    def test_get_index2(self):
        controlpoints = scipy.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        grid = scipy.array([1, 2, 3, 4, 5, 6, 7])
        spline = Bspline(grid=grid, controlpoints=controlpoints)
        self.assertEqual(spline.get_index(7), 5)

    def test_get_index3(self):
        controlpoints = scipy.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        grid = scipy.array([1, 2, 3, 4, 5, 6, 7])
        spline = Bspline(grid=grid, controlpoints=controlpoints)
        self.assertEqual(spline.get_index(1), 0)

    def test_get_controlpoints1(self):
        spline = self.get_spline()
        index = scipy.random.randint(2, len(spline.controlpoints) - 3)
        self.assertTrue((spline.get_controlpoints(index) ==
                         spline.controlpoints[index-2:index+2]).all())

    def test_get_controlpoints2(self):
        spline = self.get_spline()
        index = scipy.random.randint(0, 2)
        self.assertTrue((spline.get_controlpoints(index) ==
                         spline.controlpoints[0:4]).all())

    def test_get_controlpoints3(self):
        spline = self.get_spline()
        index = len(spline.controlpoints) - 1
        self.assertTrue((spline.get_controlpoints(index) ==
                         spline.controlpoints[-4:]).all())

    def test_get_alpha(self):
        spline = self.get_spline()
        u = spline.grid[0] + scipy.random.random() * spline.grid[1]
        index = spline.get_index(u)
        indices = [index - scipy.random.randint(-1, 3),
                   index + scipy.random.randint(1, 3)]
        self.assertEqual(spline.get_alpha(u=u, indices=indices),
                         (spline.grid[indices[1]] - u) /
                         (spline.grid[indices[1]] - spline.grid[indices[0]]))

if __name__ == '__main__':
    # controlpoints = scipy.array([[40, 17],
    #                             [20, 0],
    #                             [18, 8],
    #                             [57, -27],
    #                             [8, -77],
    #                             [-23, -65],
    #                             [-100, -15],
    #                             [-23, 7],
    #                             [-40, 20],
    #                             [-15, 10]])

    controlpoints = scipy.array([[x * scipy.cos(x), x * scipy.sin(x)]
                                for x in scipy.linspace(0, 8 * scipy.pi, 35)])

    grid = scipy.hstack((scipy.zeros(2),
                         scipy.linspace(0, 1, len(controlpoints) + 2 - 4),
                         scipy.ones(2)))
    spline = Bspline(grid=grid, controlpoints=controlpoints)
    spline.plot()

#    print(spline.get_basisfunc(3,np.array([0,0.25,0.5,0.75,1]),0,k=1))
#   unittest.main()

