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

        # stepsize as attribute
        self.h = h
        # walls as attributes
        self.westwall = westwall
        self.northwall = northwall
        self.eastwall = eastwall
        self.southwall = southwall
        # setup the matrix to store the solution
        self.umatrix = self.setup_umatrix()
        # setup the matrix of the linear system which is to be solved
        self.A = self.setup_A()
        # setup the righthand side of the linear system which is to be solved
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
        column = scipy.concatenate((scipy.array([-4, 1]),
                                    scipy.zeros(self.northwall.len - 2)))
        T = scipy.linalg.toeplitz(column)
#        Ttuple = tuple((T for _ in range(len(self.umatrix[:, 0]) - 2)))
        Ttuple = tuple((T for _ in range(self.westwall.len)))
        A = scipy.linalg.block_diag(*Ttuple) \
            + \
            scipy.eye(self.northwall.len * self.westwall.len,
                      k=self.northwall.len) \
            + \
            scipy.eye(self.northwall.len * self.westwall.len,
                      k=-self.northwall.len)
        return (1 / self.h ** 2) * A

    def setup_b(self):
        N = self.northwall.len + 2
        M = self.westwall.len + 2
        b = scipy.zeros((self.westwall.len, self.northwall.len))
        b[0, 0] = self.umatrix[0, 1] + self.umatrix[1, 0]
        b[0, -1] = self.umatrix[0, N-2] + self.umatrix[1, N-1]
        b[-1, 0] = self.umatrix[M-2, 0] + self.umatrix[M-1, 1]
        b[-1, -1] = self.umatrix[M-1, N-2] + self.umatrix[M-2, N-1]
        for i in range(1, N-3):
            b[0, i] = self.umatrix[0, i]
            b[-1, i] = self.umatrix[M-1, i]
        for i in range(1, M-3):
            b[i, 0] = self.umatrix[i, 0]
            b[i, -1] = self.umatrix[i, N-1]
        return (- (1 / self.h ** 2) * b).flatten()

    def get_solution(self):
        self.u = scipy.linalg.solve(self.A, self.b)

    def plot(self):
        self.umatrix[1:-1, 1:-1] = self.u.reshape((self.umatrix.shape[0] - 2,
                                                   self.umatrix.shape[1] - 2))
        pyplot.close()
        figure = pyplot.figure()
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
    def __init__(self, values, condition=None):
        self.len = len(values)
        self.values = values
        self.condition = condition

if __name__ == '__main__':
    h = 0.5
    westwall = wall(2 * scipy.ones(16))
    northwall = wall(0.5 * scipy.ones(15))
    eastwall = wall(scipy.ones(16))
    southwall = wall(1.5 * scipy.ones(15))
#    westwall = wall(2 * scipy.ones(6))
#    northwall = wall(2 * scipy.ones(6))
#    eastwall = wall(scipy.ones(6))
#    southwall = wall(2 * scipy.ones(6))
#    westwall = wall(scipy.zeros(8))
#    northwall = wall(scipy.zeros(8))
#    eastwall = wall(scipy.zeros(8))
#    southwall = wall(scipy.zeros(8))
    R = room(westwall=westwall,
              northwall=northwall,
              eastwall=eastwall,
              southwall=southwall,
              h=h)
#    print(R.A)
#    print(R.b)
#    print(R.u)
    R.get_solution()
    figure = R.plot()

    figure.show()
