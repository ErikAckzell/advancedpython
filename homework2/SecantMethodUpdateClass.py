from NewtonMethodClass import NewtonMethod
import scipy


class SecantMethodUpdate(NewtonMethod):
    """
    This is a general class for the Quasi Newton methods which uses a secant
    method. It inherits from the QuasiNewtonMethods class.
    """
    def __init__(self, problem, x0, linesearchoption, linesearchcondition):
        super().__init__(problem, x0, linesearchoption, linesearchcondition)
        self.x = x0
        self.x_old = scipy.copy(self.x)
        self.gk = self.g(self.x)
        self.gk_old = self.g(self.x_old)

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

    def get_alphak(self):
        self.linesearch.search(search_option=self.linesearchoption,
                               cond_option=self.linesearchcondition)
