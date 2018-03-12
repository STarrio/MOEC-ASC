"""
Algorithm module
"""

import zdt3
import random
import numpy as np

def euclidean(v1,v2):
    return np.linalg.norm(v1-v2)

class MOEC:
    def __init__(self,problem=zdt3.ZDT3(),generations,neighbourhood=0.2):
        self.n_sp = problem.n_obj
        self.lambda_vectors=[np.random.dirichlet(np.ones(self.n_sp),size=1) for _ in self.n_sp]