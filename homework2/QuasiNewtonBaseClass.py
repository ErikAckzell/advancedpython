# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 12:51:51 2016

@author: Erik Ackzell
"""


import abc


class QuasiNewtonBase(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_solution(self):
        pass

    @abc.abstractmethod
    def update_s(self):
        pass

    @abc.abstractmethod
    def update_alpha(self):
        pass

    @abc.abstractmethod
    def update_x(self):
        pass

    @abc.abstractmethod
    def update_H(self):
        pass
