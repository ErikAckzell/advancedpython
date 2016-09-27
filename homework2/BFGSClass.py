from SecantMethodUpdateClass import SecantMethodUpdate


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
