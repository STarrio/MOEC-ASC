"""
Developed by Santiago Tarrio

ZDT3 function in Python for the multiobjective evolutionary algorithm.
"""
import math

class ZDT3():
    def __init__(self,n=30):
        self.n_real = n
        self.n_obj = 2
        self.min_real = [0.0 for _ in range(n)]
        self.max_real = [1.0 for _ in range(n)]

    def func(self,xreal):
        obj_0 = xreal[0]
        g = 1+((9*sum(xreal[1:]))/(self.n_real-1))
        h = 1 - math.sqrt(obj_0/g)-(obj_0/g)*math.sin(10*math.pi*obj_0)
        obj_1 = g*h
        return (obj_0,obj_1)
