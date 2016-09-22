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

    def __init__(self, problem, dim):
        self.x = scipy.zeros(dim)
        self.x_old = scipy.ones(dim)
        self.f = problem.f
        self.g = problem.g
        self.gk = self.g(self.x)
        self.gk_old = self.g(self.x_old)
        self.H = scipy.eye(dim)

    def get_solution(self, tolerance=1e-8):
        while scipy.linalg.norm(self.g(self.x)) > tolerance:
            self.perform_iteration()
        return self.x

    def perform_iteration(self):
        sk = self.get_sk()
        alphak = self.get_alphak()
        self.update_x(sk=sk, alphak=alphak)


class NewtonMethod(QuasiNewtonBaseClass.QuasiNewtonBase,
                   QuasiNewtonMethods):

    def update_x(self, sk, alphak):
        self.x_old = scipy.copy(self.x)
        self.x = self.x_old + alphak * sk

    def update_H(self):
        pass

    def get_alphak(self):
        pass

    def get_sk(self):
        return - self.H.dot(self.g(self.x))

############# SECANT METHODS (TASK 9) #############


class SecantMethodUpdate(QuasiNewtonMethods):

    def get_delta(self):
        return self.x - self.x_old

    def get_gamma(self):
        return self.gk - self.gk_old

    def update_x(self):
        self.x_old = scipy.copy(self.x)
        self.x = self.x - self.H.dot(self.gk)


class goodBroyden(SecantMethodUpdate):

    def update_H(self):
        self.shermanMorrisonUpdate()

    def shermanMorrisonUpdate(self):
        gamma = self.get_gamma()
        delta = self.get_delta()
        u = delta - self.H.dot(gamma)
        a = 1 / u.transpose.dot(gamma)
        self.H = self.H + a * u.dot(u.transpose())


class badBroyden(SecantMethodUpdate):

    def update_H(self):
        self.bad_update()

    def bad_update(self):
        gamma = self.get_gamma()
        delta = self.get_delta()
        self.H = (self.H +
                  ((delta - self.H.dot(gamma)) /
                   scipy.linalg.norm(gamma)**2) *
                  gamma.transpose())


class DFP(SecantMethodUpdate):

    def update_H(self):
        pass

    def DFPupdate(self):
        gamma = self.get_gamma()
        delta = self.get_delta()
        summand2 = delta.dot(delta.transpose()) / delta.transpose().dot(gamma)
        summand3 = (self.H.dot(gamma).dot(gamma.transpose()).dot(self.H) /
                    gamma.transpose().dot(self.H).dot(gamma))
        self.H = self.H + summand2 + summand3


class BFGS(SecantMethodUpdate):

    def update_H(self):
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
#    unittest.main()
