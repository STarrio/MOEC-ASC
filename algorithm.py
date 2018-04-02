"""
Developed by Santiago Tarrio

Algorithm module
"""

import zdt3
import random
import numpy as np
import parsing

class MOEC:
    # Evaluaciones: N x G <= 4000 ó 10000
    # N: n_sp
    # G: generations
    def __init__(self,n_sp,generations,neighbourhood,de_F,de_CR,de_SIG,path_to_file,const_mode=None,problem=zdt3.ZDT3(),weights=False):
        self.n_sp = n_sp
        self.problem = problem
        self.generations = generations
        self.de_F = de_F
        self.de_CR = de_CR
        self.de_SIG = de_SIG
        self.const = const_mode
        self.weights = weights if weights else np.array([1 for _ in range(problem.n_con)])

        self.EP = list()

        self.lambda_vectors = [np.array([n/(self.n_sp-1),1-(n/(self.n_sp-1))]) for n in range(self.n_sp)]

        self.neighbours = {}
        for i in range(self.n_sp):
            self.neighbours[i] = self.get_neighbours(self.lambda_vectors[i],list(range(self.n_sp)),self.lambda_vectors,neighbourhood)

        self.population = [np.random.uniform(self.problem.min_real,self.problem.max_real) for _ in range(self.n_sp)]
        performances = [self.problem.func(s) for s in self.population]
        self.z_star = np.array([min([ind[o] for ind in performances]) for o in range(self.problem.n_obj)])
        self.initialize_EP(performances)
        self.parser = parsing.Parser(path_to_file)


    def get_neighbours(self,v1,sps,lambdas,T):
        return sorted(sps, key = lambda v: np.linalg.norm(v1-lambdas[v]))[:T]

    def run(self):
        for g in range(self.generations):
            for i in range(self.n_sp):
                y = self.recombination(i)
                y_performance = self.problem.func(y)
                self.update_EP(y,y_performance)

                for j in range(self.problem.n_obj):
                    if(y_performance[j]<=self.z_star[j]):
                        self.z_star[j]=y_performance[j]

                for j in self.neighbours[i]:
                    if(self.fitness(y,self.population[j],self.lambda_vectors[j])):
                        self.population[j]=y
            self.parser.write_popm(self.population,self.problem)
        print("Finished")
        return self.population

    def recombination(self,i):
        # todo el vecindario de x_j puede salir, incluyendo x_j, que PUEDE SALIR, pero no tiene por que
        # tambien vale mi version, PROBAR
        # CON REEMPLAZAMIENTO
        # incluye a las soluciones de EP
        differential_sp = np.random.choice(self.neighbours[i]+list(range(len(self.EP))),size=3,replace=False)
        vector_pool = self.population + [f[0] for f in self.EP]
        vectors = [vector_pool[v] for v in differential_sp]
        de_result = vectors[0]+self.de_F*(vectors[1]-vectors[2])

        # genero un nuevo vector donde, con una cierta probabilidad CR, se copia cada posicion del vector de
        # population[i] o de de_result

        de_result = np.array([de_result[p] if random.random()>= self.de_CR else self.population[i][p] for p in range(len(de_result))])

        #con probabilidad 1/p significa que estadisticamente solo cambiara UNO de los elementos
        sigma = lambda p: (self.problem.max_real[p]-self.problem.min_real[p])/self.de_SIG
        de_result = np.array([de_result[p] + np.random.normal(0,sigma(p)) if random.random()>= 1/self.problem.n_real else de_result[p]
                        for p in range(len(de_result))])
        for e in range(len(de_result)):
            if(de_result[e] > self.problem.max_real[e]):
                de_result[e]=self.problem.max_real[e]
            if(de_result[e] < self.problem.min_real[e]):
                de_result[e]=self.problem.min_real[e]
        return de_result

    def fitness(self,x,y,lamb):
        if not self.const:
            return self.tchebycheff(x,lamb) <= self.tchebycheff(y,lamb)
        elif(self.const == "penalty"):
            return self.tchebycheff(x,lamb) + self.penalty(x) <= self.tchebycheff(y,lamb) + self.penalty(y)
        else:
            return self.constraint_dominance(x,y,lamb)

    def tchebycheff(self,x,lamb):
        return max(lamb[i]*abs(self.problem.func(x)[i]-self.z_star[i]) for i in range(self.problem.n_obj))

    def penalty(self,x):
        v = self.problem.const(x)
        return sum(self.weights*v)

    def constraint_dominance(self,s1,s2,lamb):
        if(self.penalty(s1) < self.penalty(s2)):
            return True
        else:
            return self.tchebycheff(s1,lamb) <= self.tchebycheff(s2,lamb)

    def initialize_EP(self,performances):
        for s in range(len(self.population)):
            if(all([performances[s][0]<performances[c][0] or performances[s][1]<performances[c][1] for c in range(len(self.population)) if c!=s])):
                self.EP.append((self.population[s],performances[s]))

    def update_EP(self,y,y_p):
        self.EP = [e for e in self.EP if not (y_p[0] < e[1][0] and y_p[1] < e[1][1])]

        if(all([y_p[0]<e[1][0] or y_p[1]<e[1][1] for e in self.EP])):
            self.EP.append((y,y_p))

