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
        self.generations = generations

        #revisar
        self.lambda_vectors=[np.random.dirichlet(np.ones(self.problem.n_obj),size=1) for _ in range(self.n_sp)]
        self.neighbours = {}
        for i in range(self.n_sp):
            self.neighbours[i] = self.get_neighbours(self.lambda_vectors[i],[n for n in range(self.n_sp) if n!=i],self.lambda_vectors,neighbourhood)

        self.population = [np.random.uniform(self.problem.min_real,self.problem.max_real) for _ in range(self.n_sp)]
        performances = [self.problem.func(s) for s in self.population]
        self.z_star = np.array([min([ind[o] for ind in self.population]) for o in range(self.problem.n_obj)])


    def get_neigbours(self,v1,sps,lambdas,T):
        return sorted(lambdas, key = lambda v: np.linalg.norm(v1-lambdas[v]))[:T]

    def run(self):
        for g in range(self.generations):
            for i in range(self.n_sp):
                y = self.recombination(i)
                y_performance = self.problem.func(y)
                self.population[i] = y
                for j in self.neigbours[i]:
                    if(self.tchebycheff(y,self.lambda_vectors[j]) <= self.tchebycheff(self.population[j],self.lambda_vectors[j])):
                        self.population[j]=y
    
    def recombination(self,i):
        return 0
    
    def tchebycheff(self,x,lamb):
        return 0
