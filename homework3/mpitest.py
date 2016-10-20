# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 20:10:35 2016
@author: robintiman
"""
import numpy
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
rankF = numpy.array(float(rank))
total = numpy.zeros(1)
comm.Reduce(rankF,total, op=MPI.SUM)
if rank == 0:
    print(total)
print(rank, rankF, total)