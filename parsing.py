"""
Developed by Santiago Tarrio

Parsing module
Reads plot.out and *** from the NSGAII output, and writes data from the algorithm to gnuplot-prepared file formats

"""

import numpy as np

def read_plotout(path):
    with open(path) as f:
        read_data = f.read()
    read_data = read_data.split("\n")
    p_parsed=[(float(x.split("\t")[0]),float(x.split("\t")[1])) for x in read_data[:-1]]
    nsga_x = np.array([p[0] for p in p_parsed])
    nsga_y = np.array([p[1] for p in p_parsed])
    return (nsga_x,nsga_y)

def plot_out(results,problem,test_n):
    p = open("MOEC_outs/plot_{0}_{1}.out".format(problem.stringify(),test_n),"w+")
    for l in results:
        p.write("{0} {1}\n".format(l[0],l[1]))
    p.close()

class Parser:
    def __init__(self,path):
        self.path = "MOEC_outs/"+path
        open(self.path,"w+")
    def write_popm(self,pop,problem):
        with open(self.path,"a+") as popm:
            for p in pop:
                res = problem.func(p)
                const = sum(problem.const(p))
                popm.write("{0} {1} {2}\n".format(res[0],res[1],const))

