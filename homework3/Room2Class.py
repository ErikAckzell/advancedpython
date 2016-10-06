# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 10:41:43 2016

@author: erik
"""


import scipy


class room2:
    """
    An object of this class is a room of type 2.
    """
    def __init__(self, westwall, northwall, eastwall, southwall, h):
        """
        The walls should include the corners.
        """
        if not len(westwall) == len(eastwall):
            raise ValueError('westwall and eastwall must have same length')
        if not len(northwall) == len(southwall):
            raise ValueError('northwall and southwall must have same length')
        self.u = scipy.zeros((len(westwall), len(northwall)))
        self.u[:, 0] = westwall
        self.u[0, :] = northwall
        self.u[:, -1] = eastwall
        self.u[-1, :] = southwall
        self.h = h
        self.matrix = self.setup_matrix()
        self.rhs = self.setup_rhs()

    def setup_matrix(self):
        column = scipy.concatenate((scipy.array([-4, 1]),
                                    scipy.zeros(len(self.u[0]) - 4)))
        T = scipy.linalg.toeplitz(column)
        Ttuple = tuple((T for _ in range(len(self.u[:, 0]) - 2)))
        A = scipy.linalg.block_diag(*Ttuple) + \
            scipy.eye((len(self.u[0]) - 2) * (len(self.u[:, 0]) - 2),
                      k=len(self.u[0]) - 2) + \
            scipy.eye((len(self.u[0]) - 2) * (len(self.u[:, 0]) - 2),
                      k=-(len(self.u[0]) - 2))
        return A

    def setup_rhs(self):
        N = len(self.u[0])
        M = len(self.u[:, 0])
        rhs = scipy.zeros(((M - 2), (N - 2)))
        rhs[0, 0] = self.u[0, 1] + self.u[1, 0]
        rhs[0, -1] = self.u[0, N-2] + self.u[1, N-1]
        rhs[-1, 0] = self.u[M-2, 0] + self.u[M-1, 1]
        rhs[-1, -1] = self.u[M-1, N-2] + self.u[M-2, N-1]
        print(rhs)
        for i in range(1, N-3):
            rhs[0, i] = self.u[0, i]
            rhs[-1, i] = self.u[M-1, i]
        print(rhs)
        for i in range(1, M-3):
            rhs[i, 0] = self.u[i, 0]
            rhs[i, -1] = self.u[i, N-1]
        return (- (1 / self.h ** 2) * rhs).flatten()

if __name__ == '__main__':
    h = 0.5
    westwall = scipy.ones(6)
    northwall = scipy.ones(5)
    R = room2(westwall=westwall,
              northwall=northwall,
              eastwall=westwall,
              southwall=northwall,
              h=h)
    print(R.matrix)
    print(R.rhs)
    print(R.u)
