from SecantMethodUpdateClass import SecantMethodUpdate
import optimizationProblemclass
import scipy


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

if __name__ == '__main__':
    def f(x):
        return x[0]**2 + x[1]**2

    def g(x):
        return scipy.array([2 * x[0], 2 * x[1]])

    p = optimizationProblemclass.optimizationProblem(function=f, gradient=g)
    instance = goodBroyden(problem=p,
                           x0=scipy.ones(2),
                           linesearchoption='inexact',
                           linesearchcondition=3)
