# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 20:10:35 2016
@author: robintiman
"""
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD

'''
The rank is the process id.

Send and Recv are blocking functions, meaning that when calling Recv the program
will be idle until a message is recieved. There is Isend and Irecv for immediate
execution.

If the sender is unknown, ANY_SOURCE can be used.


'''

rank = comm.Get_rank()

rand = np.zeros(1)

if rank == 0:
	rand = np.random.random_sample(1)
	print("Process", rank, "drew the number", rand[0])
	comm.Send(rand, dest=1)

if rank == 1:
	#
	comm.Recv(rand, source=0)
	print("Process", rank, "revieved the number", rand[0])
