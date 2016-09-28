import scipy
from SecantMethodUpdateClass import SecantMethodUpdate
import optimizationProblemclass


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

if __name__ == '__main__':
    def f(x):
        return x[0]**2 + x[1]**2

    def g(x):
        return scipy.array([2 * x[0], 2 * x[1]])

    p = optimizationProblemclass.optimizationProblem(function=f, gradient=g)
    instance = badBroyden(problem=p,
                          x0=scipy.ones(2),
                          linesearchoption='inexact',
                          linesearchcondition=3)
