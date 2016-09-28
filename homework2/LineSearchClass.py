# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 20:30:55 2016
@author: robintiman
"""
import scipy as sp
import math

class LineSearch():
    INEXACT = 1
    EXACT = 2
    WOLFE = 3
    GOLDSTEIN = 4

    def __init__(self, f):
        self.f = f

    # Use this method along with the option value
    def search(self, search_option, cond_option):
        if (search_option == self.INEXACT):
            self.inexact(self, a0, al = 0, au = sys.maxint, t = 0.1, x = 9,
                         sigma = 0.7, p = 0.1, f, cond_option)
        elif (search_option == self.EXACT):
            self.exact(self.f, a0)

    def __exact(f):
        return sp.optimize.fmin(f, a0)

    def __inexact(self, a0, al, au, t, x, p, f, sigma, option):
        lc, rc = False
        while not(lc and rc):
            if lc:
                au = min(a0, au)
                a_hat = self.interpolate(a0, al, f)
                a_hat = max(a_hat, al+t*(au-al))
                a_hat = min(a_hat, au-t*(au-al))
                a0 = a_hat
            else:
                delta_a = self.extrapolate(a0, al, f)
                delta_a = max(delta_a, t*(a0-al))
                delta_a = min(delta_a, x*(a0-al))
                al = a0
                a0 = a0+delta_a
            lc, rc = self.checkCondition(self, f, a0, al, p, sigma, option)
        return a0, f(a0)

    def extrapolate(a0, al, f):
        delta_a = (a0-al)*f.g(a0)/(f.g(al)-f.g(a0))
        return delta_a

    def interpolate(a0, al, f):
        num = math.pow(a0-al, 2)*f.g(al)
        denom = 2*(f(al)-f(a0)+(a0-al)*f.g(al))
        return num/denom

    def checkCondition(self, f, a0, al, p, sigma, option):
        lc, rc = False
        # Goldstein conditions
        if option == self.GOLDSTEIN:
            lc = f(a0) >= f(al) + (1-p)*(a0-al)*f.g(al)

        # Wolfe-Powell conditions
        elif option == self.WOLFE:
            lc = f.g(a0) >= sigma*f.g(al)

        # The right condition is the same in both
        rc = f(a0) <= f(al) + p*(a0-al)*f.g(al)
        return lc, rc

