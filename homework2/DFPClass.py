from SecantMethodUpdateClass import SecantMethodUpdate


class DFP(SecantMethodUpdate):
    """
    This is a class for the DFP method. It inherits from the
    SecantMethodUpdate class.
    """

    def update_H(self):
        """
        This method updates H.
        """
        self.DFPupdate()

    def DFPupdate(self):
        """
        This method implements the DFP method.
        """
        gamma = self.get_gamma()
        delta = self.get_delta()
        summand2 = delta.dot(delta.transpose()) / delta.transpose().dot(gamma)
        summand3 = (self.H.dot(gamma).dot(gamma.transpose()).dot(self.H) /
                    gamma.transpose().dot(self.H).dot(gamma))
        self.H = self.H + summand2 + summand3


