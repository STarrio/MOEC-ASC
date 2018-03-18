"""
Developed by Santiago Tarrio

Algorithm module
"""

import zdt3
import random
import numpy as np

class MOEC:
    # Evaluaciones: N x G <= 4000 ó 10000
    # N: n_sp
    # G: generations
    def __init__(self,n_sp,generations,neighbourhood,de_F,problem=zdt3.ZDT3()):
        self.n_sp = n_sp
        self.problem = problem
        self.generations = generations
        self.de_F = de_F

        self.lambda_vectors = [[n/(self.n_sp-1),1-(n/(self.n_sp-1))] for n in range(self.n_sp)]
                            
        self.neighbours = {}
        for i in range(self.n_sp):
            self.neighbours[i] = self.get_neighbours(self.lambda_vectors[i],list(range(self.n_sp)),self.lambda_vectors,neighbourhood)

        self.population = [np.random.uniform(self.problem.min_real,self.problem.max_real) for _ in range(self.n_sp)]
        performances = [self.problem.func(s) for s in self.population]
        self.z_star = np.array([min([ind[o] for ind in self.population]) for o in range(self.problem.n_obj)])

    def get_neighbours(self,v1,sps,lambdas,T):
        return sorted(sps, key = lambda v: np.linalg.norm(v1-lambdas[v]))[:T]

    def run(self):
        for g in range(self.generations):
            for i in range(self.n_sp):
                y = self.recombination(i)
                y_performance = self.problem.func(y)

                for j in range(self.problem.n_obj):
                    if(y_performance[j]<=self.z_star[j]):
                        self.z_star[j]=y_performance[j]

                for j in self.neighbours[i]:
                    if(self.tchebycheff(y,self.lambda_vectors[j]) <= self.tchebycheff(self.population[j],self.lambda_vectors[j])):
                        self.population[j]=y
        return self.population
    
    def recombination(self,i):
        #revisar
        # todo el vecindario de x_j puede salir, incluyendo x_j, que PUEDE SALIR, pero no tiene por que
        # tambien vale mi version, PROBAR
        # CON REEMPLAZAMIENTO
        differential_sp = np.random.choice(self.neighbours[i],size=3)
        vectors = [self.population[v] for v in differential_sp]

        #revisar
        de_result = vectors[0]+self.de_F*(vectors[1]-vectors[2])

        #falta cruce
        # genero un nuevo vector donde, con una cierta probabilidad CR, se copia cada posicion del vector de 
        # population[i] o de de_result
        
        de_result = [de_result[p] if random.random()>= self.CR else self.population[i][p] for p in range(len(de_result))]

        #falta mutacion gaussiana, que con probabilidad 1/p significa que estadisticamente solo cambiara UNO de los elementos
        sigma = lambda p: (self.problem.max_real[p]-self.problem.min_real[p])/self.SIG
        de_result = [de_result[p] + np.random.normal(0,sigma(p)) if random.random()>= self.problem.n_real else de_result[p]
                        for p in range(len(de_result))]
                            
        return de_result
    
    def tchebycheff(self,x,lamb):
        return max(lamb[i]*abs(self.problem.func(x)[i]-self.z_star[i]) for i in range(self.problem.n_obj))