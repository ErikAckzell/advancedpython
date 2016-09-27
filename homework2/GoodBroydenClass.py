from SecantMethodUpdateClass import SecantMethodUpdate

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
