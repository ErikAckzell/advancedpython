# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 20:30:55 2016
@author: robintiman
"""
import scipy as sp
import numpy as np

class LineSearch():

    def __init__(self, function, grad, option):
        self.INEXACT = 1
        self.EXACT = 2
        self.function = function
        self.grad = grad
        self.option = option

    # Use this method along with the option value
    def search(option):
        if (option == self.INEXACT):
            __inexact()
        elif (option == self.EXACT):
            __exact()

    def __exact():
        sp.optimize.line_search()

    def __inexact(a, p, t, x, sigma, fna0, fnal):
        while (not)

    def checkAcceptable(f, a0, p, sigma):
        # Goldstein conditions
        fprime = findPrime(f)
        lc_gold = fa(a0) >= fa(aL) + (1-p)*(a0-aL)*fprime(aL)
        rc_gold = fa(a0) <= fa(aL) + p*(a0-aL)*fprime(aL)

        # Wolfe-Powell conditions
        lc_wolf = fprime(a0) >= sigma*fprime(aL)

        # The right condition of the Wolfe-Powell is the same as the one in Goldstein

        return (lc_gold and rc_gold and lc_wolf)


    def findPrime(f):
        x = Symbol('x')
        fprime = f.diff(x)
        fprime = lambdify(x, fprime, 'numpy')
        return fprime