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

    fig, ax = plt.subplots(figsize=(8,8))

    n = list(range(50))

    ax.set_aspect('equal', 'box')

    coords = getTurbLoc(sys.argv[1])
    x = coords[:,0]
    y = coords[:,1]
    ax.scatter(x, y, c='blue')

    x = range(50,3950,1)
    y = range(50,3950,1)
    ax.plot(x, y, c='red')

    # for i, txt in enumerate(n):
    #     ax.annotate(txt, (x[i]-20, y[i]-20))

    # coords = getTurbLoc(sys.argv[2])
    # x = coords[:,0]
    # y = coords[:,1]
    # ax.scatter(x, y, c='red')
    # for i, txt in enumerate(n):
    #     ax.annotate(txt, (x[i]+20, y[i]+20))

    plt.show()
