import unittest
import scipy

import optimizationProblemclass

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