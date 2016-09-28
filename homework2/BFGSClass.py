from SecantMethodUpdateClass import SecantMethodUpdate
import optimizationProblemclass
import scipy


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

if __name__ == '__main__':
    def f(x):
        return x[0]**2 + x[1]**2

    def g(x):
        return scipy.array([2 * x[0], 2 * x[1]])

    p = optimizationProblemclass.optimizationProblem(function=f, gradient=g)
    instance = BFGS(problem=p,
                    x0=scipy.ones(2),
                    linesearchoption='inexact',
                    linesearchcondition=3)
