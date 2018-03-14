"""
Algorithm module
"""

import zdt3
import random
import numpy as np


class MOEC:
    def __init__(self,n_sp,generations,neighbourhood,problem=zdt3.ZDT3()):
        self.n_sp = n_sp
        self.problem = problem
        #revisar
        self.lambda_vectors=[np.random.dirichlet(np.ones(self.problem.n_obj),size=1) for _ in range(self.n_sp)]
        self.neighbours = {}
        for i in range(self.n_sp):
            self.neighbours[i] = self.get_neighbours(self.lambda_vectors[i],[n for n in range(self.n_sp) if n!=i],self.lambda_vectors,neighbourhood)


    def get_neigbours(self,v1,sps,lambdas,T):
        return sorted(lambdas, key = lambda v: np.linalg.norm(v1-lambdas[v]))[:T]

