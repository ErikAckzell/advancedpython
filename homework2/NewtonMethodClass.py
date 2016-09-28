import scipy
import numpy as np
import numdifftools as nd
from QuasiNewtonBaseClass import QuasiNewtonBase


class NewtonMethod(QuasiNewtonBase):
    """
    This is a class for the classic Newton method. It inherits from the abstract base class QuasiNewtonBase
    """

    def update_x(self):
        """
        This mehod updates the solution x.
        """
        sk = self.get_sk()
        self.x = self.x + sk

    def update_H(self):
        """
        This method approximates the hessian H of the objective function by using finite differences followed by a
        symmetrizing step.
        """
        hessian = nd.Hessian(self.f)
        self.H = (hessian + np.transpose(hessian))* 1/2

    def get_alphak(self):
        """
        This method is not needed for the classical Newton method.
        """
        pass

    def get_sk(self):
        """
        This method calculates and returns the current s. To solve the linear system a solver based on Cholesky
        decomposition is used.
        """
        self.update_H()
        cho_factor = scipy.linalg.cholesky(-self.H) # this will raise an error if the matrix is not positive definite
        sk = scipy.linalg.cho_solve(cho_factor(self.x), self.g(self.x))
        return sk