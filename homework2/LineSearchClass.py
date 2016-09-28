# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 20:30:55 2016
@author: robintiman
"""
import scipy.optimize as sp
import numpy as np
import sys
from rosenbrock import rosenbrockfunction as rf
from rosenbrock import rosenbrockderivative as rd

class LineSearch():
    INEXACT = 1
    EXACT = 2
    WOLFE = 3
    GOLDSTEIN = 4

    # Use this method along with the option values
    def search(self, f, a0, search_opt, cond_opt = None, fder = None):
        if (search_opt == self.INEXACT):
            return self.inexact(f, a0, fder, cond_opt, al = np.array([0, 0]), \
                                au = np.array([100, 100]), \
                                t = 0.1, x = 9, sigma = 0.7, p = 0.1)
        elif (search_opt == self.EXACT):
            return self.exact(f, a0)

    def exact(self, f, a0):
        return sp.minimize(f, a0)

    def inexact(self, f, a0, fder, cond_opt, al, au, t, x, sigma, p):
        lc = [False] * a0.size
        rc = [False] * a0.size
        while not(all(lc) and all(rc)):
            print(lc)
            print(rc)
            if all(lc):
                au = np.minimum(a0, au)
                a_hat = self.__interpolate(a0, al, f, fder)
                a_hat = np.maximum(a_hat, al+t*(au-al))
                a_hat = np.minimum(a_hat, au-t*(au-al))
                a0 = a_hat
            else:
                delta_a = self.__extrapolate(a0, al, f, fder)
                delta_a = np.maximum(delta_a, t*(a0-al))
                delta_a = np.minimum(delta_a, x*(a0-al))
                al = a0
                a0 = a0+delta_a
            lc = self.__checkLeft(f, fder, a0, al, p, sigma, cond_opt)
            rc = self.__checkRight(f, fder, a0, al, p)
        return a0, f(a0)

    def __extrapolate(self, a0, al, f, fder):
        delta_a = (a0-al)*fder(a0)/(fder(al)-fder(a0))
        return delta_a

    def __interpolate(self, a0, al, f, fder):
        num = (a0-al)**2*fder(al)
        denom = 2*(f(al)-f(a0)+(a0-al)*fder(al))
        return num/denom

    def __checkRight(self, f, fder, a0, al, p):
        return f(a0) <= f(al) + p*(a0-al)*fder(al)

    def __checkLeft(self, f, fder, a0, al, p, sigma, cond_opt):
        # Goldstein conditions
        if cond_opt == self.GOLDSTEIN:
          return f(a0) >= f(al) + (1-p)*(a0-al)*fder(al)

        # Wolfe-Powell conditions
        else:
            return fder(a0) >= sigma*fder(al)

ls = LineSearch()
guess = np.array([1, 2])
print(ls.search(rf, guess, LineSearch.INEXACT, LineSearch.WOLFE, rd))

