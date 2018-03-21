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