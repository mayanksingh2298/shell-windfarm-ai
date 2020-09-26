"""
    to plot the output on a graph
    usage: python3 visualize_output.py <path of output file>
"""

import numpy as np
import math
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
    x = 2000
    y = 2000
    plt.scatter(x, y)
    fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
    for theta in range(0,360,5):
        newx = x + 2000*math.cos((theta * 22/7 )/180)
        newy = y + 2000*math.sin((theta * 22/7 )/180)
        plt.plot([x,newx],[y,newy])
        circle1 = plt.Circle((newx, newy), 200, color='black', fill=False)
        ax.add_artist(circle1)

    plt.show()
