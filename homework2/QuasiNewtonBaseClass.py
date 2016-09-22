# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 12:51:51 2016

@author: Erik Ackzell
"""


import abc


class QuasiNewtonBase(metaclass=abc.ABCMeta):
    """
    Abstract class for Quasi Newton methods for solving minimization problems.
    """

    @abc.abstractmethod
    def get_sk(self):
        pass

    @abc.abstractmethod
    def get_alphak(self):
        pass

    @abc.abstractmethod
    def update_x(self):
        pass

    @abc.abstractmethod
    def update_H(self):
        pass
