from scipy.optimize import minimize
# If SciPy (version 1.1 or above) is installed, then Bounds, LinearConstraint,
# and NonlinearConstraint can alternatively be imported from scipy.optimize.
import numpy as np
from cma.fitness_transformations import EvalParallel2
from scipy.optimize import Bounds
from scipy import optimize
from tqdm import tqdm,trange
import os
import sys
import multiprocessing as mp
import numpy as np
import time
from evaluate import checkConstraints, binWindResourceData, getAEP, loadPowerCurve, getTurbLoc, preProcessing
from datetime import datetime
import random
from args import make_args
from tqdm import tqdm
from utils import score, score_cma, score_pdfo, initialise_valid, initialise_periphery, min_dist_from_rest, delta_score, delta_check_constraints, fetch_movable_segments
from utils import min_dis, initialise_file
from constants import *
from datetime import datetime


# args = make_args()

#-------hyperparameters---------
NPROCS = 40
NEGATIVE_CHILDREN = 20
CHILDREN_IMPROVEMENT_THRESHOLD = 0
MASTER_IMPROVEMENT_THRESHOLD = 0
MASTER_REALLY_IMPROVEMENT_THRESHOLD = 0.01
YEARS = [2007, 2008, 2009, 2013, 2014, 2015, 2017]
# YEARS = [2015, 2017]

NEGATIVE_NON_POS_INC_THRESHOLD = 10000
#-------hyperparameters---------
#no nengative processes are reinitialized at MASTER_IMPROVEMENT_THRESHOLD
#all processes are reinitialized at MASTER_REALLY_IMPROVEMENT_THRESHOLD
#initial processes are negative  
WINNER_CSV = "data/winner.csv"
CHILDREN_DATA_ROOT = "data/children"

def save_csv(coords,path):
	f = open(path,"w")
	np.savetxt(f, coords, delimiter=',', header='x,y', comments='', fmt='%1.8f')
	f.close()

def get_back_coords(coords):
	coords = coords.reshape(-1,2)
	save_csv(coords,"data/winner_cmaes.csv")

if __name__ == '__main__':
	#define the starting point [x1,y1,x2,y2,...,x50,y50]
	x0 = getTurbLoc(sys.argv[1]).reshape(-1)
	# lis = []
	# for i in range(100):
	# 	lis.append(random.random()*3900+50)
	# x0 = np.array(lis)


	#define options
	# options={'bounds': [50,3950], 'transformation':None}
	options={'disp': True, "verbose":1} 

	#define the bounds
	lb = [50] * 100
	ub = [3950] * 100
	bounds = Bounds(lb,ub)

	#start it
	print("Started Algo")

	# res = minimize(score_pdfo, x0, method='SLSQP', bounds = bounds, options=options)
	res = optimize.shgo(score_pdfo, bounds = [(50,3950)]*100, options=options)

	print(res.x)

	import pdb
	pdb.set_trace()
