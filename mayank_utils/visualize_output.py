"""
    to plot the output on a graph
    usage: python3 visualize_output.py <path of output file>
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
def getTurbLoc(turb_loc_file_name):
    f = open(turb_loc_file_name,'r').readlines()
    l = []
    for line in f[1:]:
        line = line.strip().split(',')
        l.append([])
        l[-1].extend([float(line[0]),float(line[1])])
    turb_coords = np.array(l,dtype = np.float32)
    return(turb_coords)

if __name__=="__main__":
    N = 50
    coords = getTurbLoc(sys.argv[1])
    x = coords[:,0]
    y = coords[:,1]
    plt.scatter(x, y)
    plt.show()
