# temp.py
# from evaluate import 
import numpy as np
import pandas as pd

f = open('optimal_50.txt', 'r')

M_EPS = 1e-6

pts = []
for _ in range(50):
	a = f.readline()
	_, x, y = [float(i) for i in a.split()]
	pts.append((x,y))

pts = np.array(pts)

# pts = 2*pts
pts = pts*(1/(1-(2-M_EPS)*(0.071377103865)))
# pts = 2*pts
pts = pts + 0.5
pts = 3900*pts + 50

# pts = pts-2000
# pts = (pts - 2000)
from utils import *
from constants import *

years = [2007, 2008, 2009, 2013, 2014, 2015, 2017]
# years = [2007]
year_wise_dist = np.array([binWindResourceData('../data/WindData/wind_data_{}.csv'.format(year)) for year in years])
wind_inst_freq = np.mean(year_wise_dist, axis = 0)

# coords = initialise_valid()
# s = score(coords, wind_inst_freq, True)


# for d in range(36):
# 	dist = []
# 	for i in range(540):
# 		dist.append(wind_inst_freq[(d*15 + i)%540])
# 	score(pts, dist, True)

# f = open("submissions/max_dist.csv", "w")
# np.savetxt(f, pts, delimiter=',', header='x,y', comments='', fmt='%1.8f')
# f.close()

pts = pd.read_csv("submissions/best.csv", header = None)
pts = pts.to_numpy()[1:].astype("float")
score(pts, wind_inst_freq, True)