"""
Developed by Santiago Tarrio

CF6 with 4 dimensions function in Python for the multiobjective evolutionary algorithm.
"""

import math
import numpy as np

class CF6:
    def __init__(self,n=4):
        self.n_real = n
        self.n_obj = 2
        self.n_con = 2
        self.min_real = np.array([0.0,-2.0,-2.0,-2.0])
        self.max_real = np.array([1.0,2.0,2.0,2.0])

    def func(self,xreal):
        sum1,sum2 = 0.0
        mysign = lambda x: 1.0 if x>0 else -1.0
        for j in range(2,self.n_real):
            if(j%2==1):
                sum1+=(xreal[j-1] - 0.8*xreal[0]*math.cos(6.0*math.pi*xreal[0] + (j*math.pi/self.n_real)))**2
            else:
                sum2+=(xreal[j-1] - 0.8*xreal[0]*math.cos(6.0*math.pi*xreal[0] + (j*math.pi/self.n_real)))**2
        obj = (xreal[0]+sum1,(1.0-xreal[0])*(1.0-xreal[0])+sum2)
        # c1 = xreal[1]-0.8*xreal[0]*math.sin(6.0*xreal[0]*math.pi+(2.0*math.pi/self.n_real)) - mysign((xreal[0]-0.5)*(1.0-xreal[0]))*math.sqrt(abs((xreal[0]-0.5)*(1.0-xreal[0])))
        # c2 = xreal[3]-0.8*xreal[0]*math.sin(6.0*xreal[0]*math.pi+(4.0*math.pi/self.n_real)) - mysign(0.25*math.sqrt(1-xreal[0])-0.5*(1.0-xreal[0]))*math.sqrt(abs(0.25*math.sqrt(1-xreal[0])-0.5*(1.0-xreal[0])))
        # const = (c1,c2)
        return obj

    def const(self,xreal):
        mysign = lambda x: 1.0 if x>0 else -1.0
        c1 = xreal[1]-0.8*xreal[0]*math.sin(6.0*xreal[0]*math.pi+(2.0*math.pi/self.n_real)) - mysign((xreal[0]-0.5)*(1.0-xreal[0]))*math.sqrt(abs((xreal[0]-0.5)*(1.0-xreal[0])))
        c2 = xreal[3]-0.8*xreal[0]*math.sin(6.0*xreal[0]*math.pi+(4.0*math.pi/self.n_real)) - mysign(0.25*math.sqrt(1-xreal[0])-0.5*(1.0-xreal[0]))*math.sqrt(abs(0.25*math.sqrt(1-xreal[0])-0.5*(1.0-xreal[0])))
        return (c1,c2)