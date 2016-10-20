# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 20:30:55 2016
@author: robintiman
"""
import scipy.optimize as sp
import sys

class LineSearch():
    INEXACT = 1
    EXACT = 2
    WOLFE = 3
    GOLDSTEIN = 4

    # Use this method along with the option values
    def search(self, f, a0, search_opt, cond_opt = None, fder = None):
        if (search_opt == self.INEXACT):
            return self.inexact(f, a0, fder, cond_opt, al = 0, au = sys.maxsize,
                                t = 0.1, x = 9, sigma = 0.7, p = 0.1)
        elif (search_opt == self.EXACT):
            return self.exact(f, a0)

    def exact(self, f, a0):
        return sp.minimize(f, a0)

    def inexact(self, f, a0, fder, cond_opt, al, au, t, x, sigma, p):
        lc = False
        rc = False
        while not(lc and rc):
            if rc:
                au = min(a0, au)
                a_hat = self.__interpolate(a0, al, f, fder)
                a_hat = max(a_hat, al+t*(au-al))
                a_hat = min(a_hat, au-t*(au-al))
                a0 = a_hat
            else:
                delta_a = self.__extrapolate(a0, al, f, fder)
                delta_a = max(delta_a, t*(a0-al))
                delta_a = min(delta_a, x*(a0-al))
                al = a0
                a0 = a0+delta_a

            try:
                lc = self.__checkLeft(f, fder, a0, al, p, sigma, cond_opt)
                rc = self.__checkRight(f, fder, a0, al, p)
            except ZeroDivisionError:
                raise ZeroDivisionError("Try another starting guess")

        return a0, f(a0)

    def __extrapolate(self, a0, al, f, fder):
        return (a0-al)*fder(a0)/(fder(al)-fder(a0))

    def __interpolate(self, a0, al, f, fder):
        num = ((a0-al)**2)*fder(al)
        denom = 2*(f(al)-f(a0)+(a0-al)*fder(al))
        return num/denom

    def __checkRight(self, f, fder, a0, al, p):
        # The right condition is the same in both
        return f(a0) <= f(al) + p*(a0-al)*fder(al)

    def __checkLeft(self, f, fder, a0, al, p, sigma, cond_opt):
        # Goldstein conditions
        if cond_opt == self.GOLDSTEIN:
            return f(a0) >= f(al) + (1-p)*(a0-al)*fder(al)

        # Wolfe-Powell conditions
        return fder(a0) >= sigma*fder(al)

ls = LineSearch()
guess = 1
func = lambda x: x**3 + 4*x**2 + 3
func_der = lambda x: 3*x**2 + 8*x
print(ls.search(func, guess, LineSearch.EXACT))
print(ls.search(func, guess, LineSearch.INEXACT, LineSearch.WOLFE, func_der))

