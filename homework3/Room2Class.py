# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 10:41:43 2016

@author: erik
"""


import scipy
from matplotlib import pyplot
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D


class room:
    """
    An object of the class is a rectangular room, on which the Laplace equation
    is to be solved, using Dirichlet conditions on each boundary.
    """
    def __init__(self, westwall, northwall, eastwall, southwall, h):
        """
        An object of the class is initialized with four walls and a grid
        stepsize in both x and y direction.
        """
        #  check that opposite walls are of the same size
        if not westwall.len == eastwall.len:
            raise ValueError('westwall and eastwall must have same length')
        if not northwall.len == southwall.len:
            raise ValueError('northwall and southwall must have same length')

        #  stepsize as attribute
        self.h = h
        #  walls as attributes
        self.westwall = westwall
        self.northwall = northwall
        self.eastwall = eastwall
        self.southwall = southwall
        #  setup the matrix to store the solution
        self.umatrix = self.setup_umatrix()
        #  setup the matrix of the linear system which is to be solved
        self.A = self.setup_A()
        #  setup the righthand side of the linear system which is to be solved
        self.b = self.setup_b()
        self.u = self.umatrix[1:-1, 1:-1].flatten()

    def setup_umatrix(self):
        """
        Setup the matrix to hold the solution.
        """
        #  initialize matrix to hold solution
        umatrix = scipy.zeros((self.westwall.len + 2, self.northwall.len + 2))

        #  insert the values of the walls
        umatrix[1:-1, 0] = self.westwall.values
        umatrix[0, 1:-1] = self.northwall.values
        umatrix[1:-1, -1] = self.eastwall.values
        umatrix[-1, 1:-1] = self.southwall.values

        #  average values for the corners
        umatrix[0, 0] = (umatrix[0, 1] + umatrix[1, 0]) / 2
        umatrix[0, -1] = (umatrix[0, -2] + umatrix[1, -1]) / 2
        umatrix[-1, 0] = (umatrix[-2, 0] + umatrix[-1, 1]) / 2
        umatrix[-1, -1] = (umatrix[-2, -1] + umatrix[-1, -2]) / 2

        return umatrix

    def setup_A(self):
        """
        Setup the matrix of the linear system with Dirichlet boundary
        conditions to be solved.
        The matrix is on the form
          A = [
               T I 0 0 0 ... 0 0 0
               I T I 0 0 ... 0 0 0
               0 I T I 0 ... 0 0 0
               ...
               0 0 0 0 0 ... 0 I T
              ]
        where
          T = [
               -4 1 0 0 0 ... 0 0 0
               1 -4 1 0 0 ... 0 0 0
               0 1 -4 1 0 ... 0 0 0
               ...
               0 0 0 0 0  ... 0 1 -4
              ]
        and I is the identity matrix.
        """
        #  first column of the toeplitz matrices that make up the diagonal of
        #  the matrix
        column = scipy.concatenate((scipy.array([-4, 1]),
                                    scipy.zeros(self.northwall.len - 2)))
        #  set up toeplitz matrix that make up the block iagonal of the matrix
        T = scipy.linalg.toeplitz(column)
        #  tuple of toeplitz matrices to
        Ttuple = tuple((T for _ in range(self.westwall.len)))
        #  set up matrix, using T and inserting ones of the identity matrices
        A = scipy.linalg.block_diag(*Ttuple) \
            + \
            scipy.eye(self.northwall.len * self.westwall.len,
                      k=self.northwall.len) \
            + \
            scipy.eye(self.northwall.len * self.westwall.len,
                      k=-self.northwall.len)
        return (1 / self.h ** 2) * A

    def setup_b(self):
        """
        Method to set up the vector on righthand side of the linear system to
        be solved.
        """
        #  initialize vector with zeros
        b = scipy.zeros((self.westwall.len, self.northwall.len))
        #  input values
        self.setup_b_corners(b)
        self.setup_b_north_south(b)
        for i in range(1, self.westwall.len - 1):
            b[i, 0] = self.umatrix[i, 0]
            b[i, -1] = self.umatrix[i, self.northwall.len + 1]
        return (- (1 / self.h ** 2) * b).flatten()

    def setup_b_corners(self, b):
        if self.westwall.condition == 'Dirichlet':
            b[0, 0] = self.umatrix[0, 1] + self.umatrix[1, 0]
            b[-1, 0] = self.umatrix[self.westwall.len, 0] +\
                self.umatrix[self.westwall.len + 1, 1]
        if self.eastwall.condition == 'Dirichlet':
            b[0, -1] = self.umatrix[0, self.northwall.len] +\
                self.umatrix[1, self.northwall.len + 1]
            b[-1, -1] = self.umatrix[self.westwall.len + 1,
                                     self.northwall.len] +\
                self.umatrix[self.westwall.len, self.northwall.len + 1]

    def setup_b_north_south(self, b):
        for i in range(1, self.northwall.len - 1):
            b[0, i] = self.umatrix[0, i]
            b[-1, i] = self.umatrix[self.westwall.len + 1, i]

    def get_solution(self):
        """
        Solve the linear equation system.
        """
        self.u = scipy.linalg.solve(self.A, self.b)

    def plot(self):
        """
        Returns a figure object with a plot of the solution.
        """
        #  insert the solution vector into the matrix
        self.umatrix[1:-1, 1:-1] = self.u.reshape((self.umatrix.shape[0] - 2,
                                                   self.umatrix.shape[1] - 2))
        figure = pyplot.figure(scipy.random.randint(1, 1000))
        axis = pyplot.subplot(111, projection='3d')
        x = scipy.arange(0, self.h * (self.northwall.len + 2), self.h)
        y = scipy.arange(0, self.h * (self.westwall.len + 2), self.h)
        X, Y = scipy.meshgrid(x, y)
        Z = self.umatrix
        axis.plot_wireframe(X, Y, Z)
        return figure


class wall:
    """
    This is a class for walls.
    """
    def __init__(self, values, condition='Dirichlet'):
        self.len = len(values)
        self.values = values
        self.condition = condition

if __name__ == '__main__':
    h = 0.5
    westwalls = [wall(scipy.ones(16)),
                 wall(2 * scipy.ones(13)),
                 wall(scipy.linspace(0, 10, 14)),
                 wall(scipy.array([x * scipy.sin(x)
                                   for x in scipy.linspace(0,
                                                           2 * scipy.pi,
                                                           50)]))]

    northwalls = [wall(scipy.ones(10)),
                  wall(0.5 * scipy.ones(15)),
                  wall(scipy.linspace(0, 10, 14)[::-1]),
                  wall(scipy.array([scipy.cos(x)
                                   for x in scipy.linspace(0,
                                                           2 * scipy.pi,
                                                           40)]))]

    eastwalls = [wall(scipy.ones(16)),
                 wall(scipy.array([2 * x * scipy.sin(x)
                                   for x in scipy.linspace(0,
                                                           2 * scipy.pi,
                                                           13)])),
                 wall(2 * scipy.ones(14)),
                 wall(scipy.linspace(0, 10, 50))]

    southwalls = [wall(scipy.ones(10)),
                  wall(scipy.linspace(0, 10, 15)[::-1]),
                  wall(0.5 * scipy.ones(14)),
                  wall(scipy.array([2 * scipy.sin(x)
                                   for x in scipy.linspace(0,
                                                           2 * scipy.pi,
                                                           40)]))]

    pyplot.close('all')
    for i in range(len(southwalls)):
        westwall = westwalls[i]
        northwall = northwalls[i]
        eastwall = eastwalls[i]
        southwall = southwalls[i]
        R = room(westwall=westwall,
                 northwall=northwall,
                 eastwall=eastwall,
                 southwall=southwall,
                 h=h)
        R.get_solution()
        figure = R.plot()

        figure.show()
