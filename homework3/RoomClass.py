import scipy
from matplotlib import pyplot
import scipy.linalg
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpi4py import MPI
from transfer import transfer


class room:
    """
    An object of the class is a rectangular room, on which the Laplace equation
    is to be solved, using Dirichlet conditions on each boundary.
    """
    def __init__(self, westwall, northwall, eastwall, southwall, h):
        """
        An object of the class is initialized with four walls and a grid
        stepsize in both x and y direction.
        """
        #  check that opposite walls are of the same size
        if not westwall.len == eastwall.len:
            raise ValueError('westwall and eastwall must have same length')
        if not northwall.len == southwall.len:
            raise ValueError('northwall and southwall must have same length')

        #  stepsize as attribute
        self.h = h
        #  walls as attributes
        self.westwall = westwall
        self.northwall = northwall
        self.eastwall = eastwall
        self.southwall = southwall
        #  setup the matrix to store the solution
        self.umatrix = self.setup_umatrix()
        #  setup the matrix of the linear system which is to be solved
        self.A = self.setup_A()
        #  setup the righthand side of the linear system which is to be solved
        self.b = self.setup_b()
        self.u = self.umatrix[1:-1, 1:-1].flatten()

    def setup_D(self, dim):
        """
        Setup the operator D, for central differences in 1D.
        d^2/dx^2 u = Du where D is
        -2 1 0 ... 0
        1 -2 1 ... 0
        ...
        0 ... 0 1 -2
        :dim: dimension
        :return: Toeplitz matrix of given dimension.
        """
        c = scipy.zeros(dim)
        c[:2] = [-2,1]
        return scipy.linalg.toeplitz(c)

    def setup_umatrix(self):
        """
        Setup the matrix to hold the solution.
        """
        #  initialize matrix to hold solution
        umatrix = scipy.zeros((self.westwall.len + 2, self.northwall.len + 2))

        #  insert the values of the walls, which do not include the corners(or u_0 and u_N)
        umatrix[1:-1, 0] = self.westwall.values
        umatrix[0, 1:-1] = self.northwall.values
        umatrix[1:-1, -1] = self.eastwall.values
        umatrix[-1, 1:-1] = self.southwall.values

        #  average values for the corners
        umatrix[0, 0] = (umatrix[0, 1] + umatrix[1, 0]) / 2
        umatrix[0, -1] = (umatrix[0, -2] + umatrix[1, -1]) / 2
        umatrix[-1, 0] = (umatrix[-2, 0] + umatrix[-1, 1]) / 2
        umatrix[-1, -1] = (umatrix[-2, -1] + umatrix[-1, -2]) / 2

        return umatrix

    def setup_A(self):
        """
        Creates the matrix D(*)I + I(*)D for the given dimension and boundary conditions, where (*) is the kronecker
        product. The north and south walls are expected to have Dirichlet BC. If BC for left or right is
        - 'Dirichlet': no extra rows are added, we have u_1,...,u_N-1
        - 'Neumann': an extra row is added for each
        :return: matrix A
        """
        x_dim = self.eastwall.len
        y_dim = self.northwall.len
        Dx, Ix = self.setup_D(x_dim), np.eye(x_dim)
        Dy, Iy = self.setup_D(y_dim), np.eye(y_dim)
        A = np.kron(Dx,Iy) + np.kron(Ix,Dy)

        if (self.westwall.condition == 'Neumann' or self.eastwall.condition == 'Neumann'):
            num_extra_var = self.eastwall.len  # eastwall does not include the u_0N and u_NN in the linear system
            A_dim = A.shape
            A_new = scipy.zeros((A_dim[0] + num_extra_var, A_dim[1] + num_extra_var))
            A_new[:A_dim[0], :A_dim[1]] = A

            # Correct the extra rows and columns in the matrix
            if self.eastwall.condition == 'Neumann':
                for i in range(num_extra_var):
                    # setup the extra columns
                    A_new[(i+1) * num_extra_var - 1, - num_extra_var:] = np.roll(np.eye(1,num_extra_var), i)
                    # setup right down corner
                    A_new[- num_extra_var:, - num_extra_var:] = np.eye(num_extra_var)
                    # setup the extra rows
                    A_new[- num_extra_var + i, num_extra_var*(i+1) - 1] = - 1

            elif self.westwall.condition == 'Neumann':
                for i in range(num_extra_var):
                    # setup the extra columns
                    A_new[num_extra_var * i, - num_extra_var:] = np.roll(np.eye(1,num_extra_var), i)
                    # setup right down corner
                    A_new[- num_extra_var:, - num_extra_var:] = - np.eye(num_extra_var)
                    # setup the extra rows
                    A_new[- num_extra_var + i, num_extra_var*i] = 1

            return (1 / self.h ** 2) * A_new
        else:
            return (1 / self.h ** 2) * A

    def setup_b(self):
        """
        Method to set up the vector on right hand side of the linear system to
        be solved.
        """
        #  initialize vector with zeros
        b = scipy.zeros((self.westwall.len, self.northwall.len))
        #  input values
        self.setup_b_corners(b)
        self.setup_b_north_south(b)
        self.setup_b_west_east(b)

        if self.westwall.condition == 'Neumann':
            b = - (1 / self.h ** 2) * b
            # add the extra variables as a row
            b = np.vstack((b, self.h * self.westwall.values))
            return b.flatten()

        elif self.eastwall.condition == 'Neumann':
            b = - (1 / self.h ** 2) * b
            # add the extra variables as a row
            b = np.vstack((b, self.h * self.eastwall.values))
            return b.flatten()

        else: # Dirichlet on both sides
            return (- (1 / self.h ** 2) * b).flatten()

    def setup_b_corners(self, b):
        if self.westwall.condition == 'Neumann':
            b[0,-1] = self.northwall.values[-1] + self.eastwall.values[0]
            b[-1,-1] = self.southwall.values[-1] + self.eastwall.values[-1]
        elif self.eastwall.condition == 'Neumann':
            b[0,0] = self.northwall.values[0] + self.westwall.values[0]
            b[-1,-2] = self.southwall.values[0] + self.westwall.values[-1]
        else: # Dirichlet on both eastwall and westwall
            b[0, 0] = self.umatrix[0, 1] + self.umatrix[1, 0]
            b[-1, 0] = self.umatrix[self.westwall.len, 0] + \
                       self.umatrix[self.westwall.len + 1, 1]
            b[0, -1] = self.umatrix[0, self.northwall.len] + \
                       self.umatrix[1, self.northwall.len + 1]
            b[-1, -1] = self.umatrix[self.westwall.len + 1,
                                     self.northwall.len] + \
                        self.umatrix[self.westwall.len, self.northwall.len + 1]

    def setup_b_north_south(self, b):
        if self.westwall.condition == 'Neumann':
            # northwall
            b[0,:-1] = self.northwall.values[:-1]
            # southwall
            b[-1,:-1] = self.southwall.values[:-1]
        elif self.eastwall.condition == 'Neumann':
            # northwall
            b[0,1:] = self.northwall.values[1:]
            # southwall
            b[-1,1:] = self.southwall.values[1:]
        else:
            for i in range(1, self.northwall.len - 1):
                b[0, i] = self.umatrix[0, i]
                b[-1, i] = self.umatrix[self.westwall.len + 1, i]

    def setup_b_west_east(self, b):
        if self.westwall.condition == 'Neumann':
            # eastwall
            b[1:-1,-1] = self.eastwall.values[1:-1]
        elif self.eastwall.condition == 'Neumann':
            # westwall
            b[1:-1,0] = self.westwall.values[1:-1]
        else:
            for i in range(1, self.westwall.len - 1):
                b[i, 0] = self.umatrix[i, 0]
                b[i, -1] = self.umatrix[i, self.northwall.len + 1]

    def get_solution(self):
        """
        Solve the linear equation system.
        """
        self.u = scipy.linalg.solve(self.A, self.b)
        if self.eastwall.condition == 'Neumann' or self.westwall.condition == 'Neumann':
            self.u = self.u[: - self.westwall.len]
            neu_var = self.u[- self.westwall.len:]
            self.update_umatrix(neu_var)
        else:
            self.update_umatrix()

    def update_umatrix(self,new_dir_val=None):
        """
        Updates the umatrix with the solution.
        :param new_dir_val: The extra variables when one wall has neumann condition,
        the values are the new DC for the next room
        :return: -
        """
        #  insert the solution vector into the umatrix
        self.umatrix[1:-1, 1:-1] = self.u.reshape((self.umatrix.shape[0] - 2,
                                                   self.umatrix.shape[1] - 2))
        if self.eastwall.condition == 'Neumann':
            self.umatrix[1:-1,-1] = new_dir_val
            #average the values in the corners
            self.umatrix[0,-1] = (self.umatrix[0,-2] + self.umatrix[1,-1])/2
            self.umatrix[-1,-1] = (self.umatrix[-1,-2] + self.umatrix[-2,-1])/2
        elif self.westwall.condition == 'Neumann':
            self.umatrix[1:-1, 0] = new_dir_val
            #average the values in the corners
            self.umatrix[0,0] = (self.umatrix[0,1] + self.umatrix[1,0])/2
            self.umatrix[-1,0] = (self.umatrix[-2,0] + self.umatrix[-1,1])/2

    def plot(self, title, N=None):
        """
        Returns a figure object with a plot of the solution.
        """
        figure = pyplot.figure(scipy.random.randint(1, 1000))
        axis = pyplot.subplot(111, projection='3d')
        x = scipy.linspace(0, self.h * (self.northwall.len + 2), self.northwall.len + 2)
        y = scipy.linspace(0, self.h * (self.westwall.len + 2), self.westwall.len + 2)
        X, Y = scipy.meshgrid(x, y)
        Z = self.umatrix
        axis.plot_wireframe(X, Y, Z)
        axis.set_title(title)
        axis.set_xlabel('North-South')
        axis.set_ylabel('West-East')
        axis.set_zlabel('Temperature Celcius')
        return figure


class wall:
    """
    This is a class for walls.
    """
    def __init__(self, values, condition='Dirichlet'):

        self.len = len(values)
        self.values = values
        self.condition = condition

def test_dir():
    h = 0.5
    westwalls = [wall(scipy.ones(16)),
                 wall(2 * scipy.ones(13)),
                 wall(scipy.linspace(0, 10, 14)),
                 wall(scipy.array([x * scipy.sin(x)
                                   for x in scipy.linspace(0,
                                                           2 * scipy.pi,
                                                           50)]))]

    northwalls = [wall(scipy.ones(10)),
                  wall(0.5 * scipy.ones(15)),
                  wall(scipy.linspace(0, 10, 14)[::-1]),
                  wall(scipy.array([scipy.cos(x)
                                    for x in scipy.linspace(0,
                                                            2 * scipy.pi,
                                                            40)]))]

    eastwalls = [wall(scipy.ones(16)),
                 wall(scipy.array([2 * x * scipy.sin(x)
                                   for x in scipy.linspace(0,
                                                           2 * scipy.pi,
                                                           13)])),
                 wall(2 * scipy.ones(14)),
                 wall(scipy.linspace(0, 10, 50))]

    southwalls = [wall(scipy.ones(10)),
                  wall(scipy.linspace(0, 10, 15)[::-1]),
                  wall(0.5 * scipy.ones(14)),
                  wall(scipy.array([2 * scipy.sin(x)
                                    for x in scipy.linspace(0,
                                                            2 * scipy.pi,
                                                            40)]))]

    pyplot.close('all')
    for i in range(len(southwalls)):
        westwall = westwalls[i]
        northwall = northwalls[i]
        eastwall = eastwalls[i]
        southwall = southwalls[i]
        R = room(westwall=westwall,
                 northwall=northwall,
                 eastwall=eastwall,
                 southwall=southwall,
                 h=h)
        R.get_solution()
        figure = R.plot('Room {}'.format(i))
        pyplot.show()

def test_neu():
    h = 0.5
    N=15
    westwalls = [wall(scipy.ones(N), condition='Neumann'),
                 wall(2 * scipy.ones(N)),
                 wall(scipy.linspace(0, 10, N), condition='Neumann'),
                 wall(scipy.array([x * scipy.sin(x)
                                   for x in scipy.linspace(0,
                                                           2 * scipy.pi,
                                                           N)]))]

    northwalls = [wall(scipy.ones(N)),
                  wall(0.5 * scipy.ones(N)),
                  wall(scipy.linspace(0, 10, N)[::-1]),
                  wall(scipy.array([scipy.cos(x)
                                    for x in scipy.linspace(0,
                                                            2 * scipy.pi,
                                                            N)]))]

    eastwalls = [wall(scipy.ones(N)),
                 wall(scipy.array([2 * x * scipy.sin(x)
                                   for x in scipy.linspace(0,
                                                           2 * scipy.pi,
                                                           N)]), condition='Neumann'),
                 wall(2 * scipy.ones(N)),
                 wall(scipy.linspace(0, 10, N), condition='Neumann')]

    southwalls = [wall(scipy.ones(N)),
                  wall(scipy.linspace(0, 10, N)[::-1]),
                  wall(0.5 * scipy.ones(N)),
                  wall(scipy.array([2 * scipy.sin(x)
                                    for x in scipy.linspace(0,
                                                            2 * scipy.pi,
                                                            N)]))]

    pyplot.close('all')
    for i in range(len(southwalls)):
        westwall = westwalls[i]
        northwall = northwalls[i]
        eastwall = eastwalls[i]
        southwall = southwalls[i]
        R = room(westwall=westwall,
                 northwall=northwall,
                 eastwall=eastwall,
                 southwall=southwall,
                 h=h)
        R.get_solution()
        figure = R.plot('Room {}'.format(i))
        pyplot.show()

if __name__ == '__main__':
    #test_dir()
    #test_neu()
    h = 1/20
    omega = 0.8
    N = 10 # u_0,..,u_N
    initial = np.zeros(N)
    numIterations = 1
    #Initial values
    gammaL = 15*np.ones(N)
    gammaR = 15*np.ones(N)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    trans = transfer()
    for i in range(numIterations):
        if rank == 1:
            
            if i > 0:
                trans.recieve_values(gammaL, 'east')
                trans.recieve_values(gammaR, 'west')
                
            room2 = room(northwall=wall(40*scipy.ones(N)),
                         southwall=wall(5*scipy.ones(N)),
                         westwall=wall(scipy.hstack((15*scipy.ones(N), gammaL))),
                         eastwall=wall(scipy.hstack((gammaR, 15*scipy.ones(N)))),
                         h=h)
            room2_old = room2.umatrix

            # solve the system and update umatrix
            room2.get_solution()

            # calculate the derivative for the values that are sent to next room
            gammaL = (room2.umatrix[N+1:-1,0] + room2.umatrix[N+1:-1,0]) / room2.h
            gammaR = (room2.umatrix[1:N+1,-1] + room2.umatrix[1:N+1,-2]) / room2.h
            gammaL = np.ascontiguousarray(gammaL, dtype=np.float32)
            gammaR = np.ascontiguousarray(gammaR, dtype=np.float32)
            trans.send_values(gammaL, 'east')
            trans.send_values(gammaR, 'west')

            # plot the solution
            room2.plot('Room 2, iteration {}'.format(i))

        if rank == 0:
            trans.recieve_values(gammaL)
            room1 = room(northwall=wall(15*scipy.ones(N)),
                         southwall=wall(15*scipy.ones(N)),
                         westwall=wall(40*scipy.ones(N)),
                         eastwall=wall(gammaL, condition='Neumann'),
                         h=h)
            room1_old = room1.umatrix

            # solve the system and update umatrix
            room1.get_solution()
            gammaL = room1.umatrix[1:N+1,-1]
            print(gammaL.flags)
            gammaL = np.ascontiguousarray(gammaL, dtype=np.float32)
            print(gammaL.flags)
            trans.send_values(gammaL)

            # plot the solution
            room1.plot('Room 1, iteration {}'.format(i))

        if rank == 2:
            trans.recieve_values(gammaR)
            room3 = room(northwall=wall(15*scipy.ones(N)),
                         southwall=wall(15*scipy.ones(N)),
                         westwall=wall(gammaR, condition='Neumann'),
                         eastwall=wall(40*scipy.ones(N)),
                         h=h)
            room3_old = room3.umatrix

            # solve the system and update umatrix
            room3.get_solution()
            gammaR = room3.umatrix[1:N+1,0]
            gammaR = np.ascontiguousarray(gammaR, dtype=np.float32)
            trans.send_values(gammaR)
            # plot the solution
            room3.plot('Room 3, iteration {}'.format(i))

        # Relaxation: u^k+1 = omega * u^k+1 + (1 - omega) * u^k
        # u = omega * u + (1 - omega) * u_old
        # u_old = u
    pyplot.show()

