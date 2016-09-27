import scipy
from SecantMethodUpdateClass import SecantMethodUpdate


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

