import scipy
from QuasiNewtonBaseClass import QuasiNewtonBase


class NewtonMethod(QuasiNewtonBase):
    """
    This is a class for the classic Newton method. It inherits from both
    the QuasiNewtonMethods class and the abstract base class
    QuasiNewtonBase
    """
    def update_x(self):
        """
        This mehod updates the solution x.
        """
        sk = self.get_sk()
        alphak = self.get_alphak()
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
