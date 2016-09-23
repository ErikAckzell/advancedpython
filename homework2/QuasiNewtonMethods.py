# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 15:12:35 2016

@author: Erik Ackzell
"""


import optimizationProblemclass
import QuasiNewtonBaseClass
import chebyquad_problem
import scipy
import unittest


class QuasiNewtonMethods:
    """
    This is a general class for Quasi Newton methods. It contains methods
    which are the same for all Quasi Newton methods.
    """

    def __init__(self, problem, x0):
        """
        An object of the class is initialized by an object of the
        optimizationProblem class problem, and an initial guess for the
        solution x0.
        """
        self.x = scipy.zeros_like(x0)
        self.x_old = x0
        self.f = problem.f
        self.g = problem.g
        self.gk = self.g(self.x)
        self.gk_old = self.g(x0)
        self.H = scipy.eye(len(x0))

    def get_solution(self, tolerance=1e-8, maxiter=200):
        """
        This method calculates and returns the solution x when g(x) < tolerance
        or stops the calculation when a maximum number of iterations has been
        performed.
        """
        i = 1
        while scipy.linalg.norm(self.g(self.x)) >= tolerance and i <= maxiter:
            self.perform_iteration()
            i += 1
        return self.x

    def perform_iteration(self):
        """
        This method performs one iteration of the Quasi Newton method.
        """
        sk = self.get_sk()
        alphak = self.get_alphak()
        self.update_x(sk=sk, alphak=alphak)


class NewtonMethod(QuasiNewtonBaseClass.QuasiNewtonBase,
                   QuasiNewtonMethods):
    """
    This is a class for the classic Newton method. It inherits from both
    the QuasiNewtonMethods class and the abstract base class
    QuasiNewtonBase
    """
    def update_x(self, sk, alphak):
        """
        This mehod updates the solution x, given an s and an alpha.
        """
        self.x_old = scipy.copy(self.x)
        self.x = self.x_old + alphak * sk

    def update_H(self):
        """
        This method updates H, which is an approximation of the inverse of the
        Hessian of the objective function.
        """
        pass

    def get_alphak(self):
        """
        This method calculates and returns the current alpha.
        """
        pass

    def get_sk(self):
        """
        This method calculates and returns the current s.
        """
        return - self.H.dot(self.g(self.x))


class SecantMethodUpdate(QuasiNewtonMethods):
    """
    This is a general class for the Quasi Newton methods which uses a secant
    method. It inherits from the QuasiNewtonMethods class.
    """

    def get_delta(self):
        """
        This method calculates and returns the current delta, which is the
        difference of the old and the current solution guess.
        """
        return self.x - self.x_old

    def get_gamma(self):
        """
        This method calculates and returns the current gamma, which is the
        difference of the gradient evaluated in the old and the current
        solution guess.
        """
        return self.gk - self.gk_old

    def update_x(self):
        """
        This method updates the solution guess.
        """
        self.x_old = scipy.copy(self.x)
        self.x = self.x - self.H.dot(self.gk)


class goodBroyden(SecantMethodUpdate):
    """
    This is a class for the "good" Broyden method using the Sherman Morrison
    update for approximating the inverse of the Hessian of the objective
    function. It inherents from the SecantMethodUpdate class.
    """
    def update_H(self):
        """
        This method updates H.
        """
        self.shermanMorrisonUpdate()

    def shermanMorrisonUpdate(self):
        """
        This method implements the "good" Broyden method.
        """
        gamma = self.get_gamma()
        delta = self.get_delta()
        u = delta - self.H.dot(gamma)
        a = 1 / u.transpose.dot(gamma)
        self.H = self.H + a * u.dot(u.transpose())


class badBroyden(SecantMethodUpdate):
    """
    This is a class for the "bad" Broyden method. It inherits from the
    SecantMethodUpdate class.
    """

    def update_H(self):
        """
        This method updates H.
        """
        self.bad_update()

    def bad_update(self):
        """
        This method implements the "bad" Broyden method.
        """
        gamma = self.get_gamma()
        delta = self.get_delta()
        self.H = (self.H +
                  ((delta - self.H.dot(gamma)) /
                   scipy.linalg.norm(gamma)**2) *
                  gamma.transpose())


class DFP(SecantMethodUpdate):
    """
    This is a class for the DFP method. It inherits from the
    SecantMethodUpdate class.
    """

    def update_H(self):
        """
        This method updates H.
        """
        self.DFPupdate()

    def DFPupdate(self):
        """
        This method implements the DFP method.
        """
        gamma = self.get_gamma()
        delta = self.get_delta()
        summand2 = delta.dot(delta.transpose()) / delta.transpose().dot(gamma)
        summand3 = (self.H.dot(gamma).dot(gamma.transpose()).dot(self.H) /
                    gamma.transpose().dot(self.H).dot(gamma))
        self.H = self.H + summand2 + summand3


class BFGS(SecantMethodUpdate):
    """
    This is a class for the BFGS method. It inherits from the
    SecantMethodUpdate class.
    """

    def update_H(self):
        """
        This method updates H.
        """
        gamma = self.get_gamma()
        delta = self.get_delta()
        summand2 = ((1 + (gamma.transpose().dot(self.H).dot(gamma) /
                          delta.transpose().dot(gamma))) *
                    delta.dot(delta.transpose()) / delta.transpose().dot(gamma)
                    )
        summand3 = - ((delta.dot(gamma.transpose()).dot(self.H) +
                       self.H.dot(gamma).dot(delta.transpose())) /
                      delta.transpose().dot(gamma))
        self.H = self.H + summand2 + summand3


class TestOptimization(unittest.TestCase):

    def test_chebyshev(self):
        problem = optimizationProblemclass.optimizationProblem(
                        function=chebyquad_problem.chebyquad,
                        gradient=chebyquad_problem.gradchebyquad)
        for method in [NewtonMethod]:
            for n in [4, 8, 11]:
                x = scipy.random.random(n)
                instance = method(problem=problem, dim=n+1)
                self.assertAlmostEqual(scipy.optimize.fmin_bfgs(
                                           chebyquad_problem.chebyquad,
                                           x,
                                           chebyquad_problem.gradchebyquad,
                                           gtol=1e-8),
                                       instance.get_solution(1e-8))

if __name__ == '__main__':
    # unittest.main()
    def f(x): return x[0] + x[1]

    def g(x): return scipy.array([1, 1])

    p = optimizationProblemclass.optimizationProblem(function=f, gradient=g)
    instance = NewtonMethod(p, 2)
    instance = goodBroyden(p, 2)
    instance = badBroyden(p, 2)
    instance = DFP(p, 2)
    instance = BFGS(p, 2)
