import abc
import scipy
from LineSearchClass import LineSearch


class QuasiNewtonBase(metaclass=abc.ABCMeta):
    """
    Abstract class for Quasi Newton methods for solving minimization problems.
    """

    def __init__(self, problem, x0,
                 linesearchoption=None, linesearchcondition=None):
        """
        An object of the class is initialized by an object of the
        optimizationProblem class problem, and an initial guess for the
        solution x0.
        """
        self.x = x0
        self.f = problem.f
        self.g = problem.g
        self.H = scipy.eye(len(x0))
        self.linesearch = LineSearch(self.f)
        self.linesearchmethod = linesearchoption
        self.linesearchcondition = linesearchcondition

    @abc.abstractmethod
    def get_sk(self):
        pass

    @abc.abstractmethod
    def get_alphak(self):
        pass

    @abc.abstractmethod
    def update_x(self):
        pass

    @abc.abstractmethod
    def update_H(self):
        pass

    def get_solution(self, tolerance=1e-8, maxiter=200):
        """
        This method calculates and returns the solution x when g(x) < tolerance
        or stops the calculation when a maximum number of iterations has been
        performed.
        """
        i = 1
        while scipy.linalg.norm(self.g(self.x)) >= tolerance and i <= maxiter:
            self.update_x()
            i += 1
        return self.x
