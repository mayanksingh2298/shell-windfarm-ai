from cma import CMAEvolutionStrategy
# If SciPy (version 1.1 or above) is installed, then Bounds, LinearConstraint,
# and NonlinearConstraint can alternatively be imported from scipy.optimize.
import numpy as np
from cma.fitness_transformations import EvalParallel2

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
from utils import score, score_cma, initialise_valid, initialise_periphery, min_dist_from_rest, delta_score, delta_check_constraints, fetch_movable_segments
from utils import min_dis, initialise_file
from constants import *
from datetime import datetime


# args = make_args()

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

	sigma0 = 0.005

	#define options
	options={'bounds': [50,3950], 'transformation':None}


	#define the bounds
	# lb = [50] * 100
	# ub = [3950] * 100
	# bounds = Bounds(lb,ub)

	#start it
	print("Started CMA")

	es = CMAEvolutionStrategy(x0, sigma0, options)
	# es.optimize(score_cma)
	# res = es.result
	es.opts.set({'tolfacupx': 1e999999, 'tolflatfitness':1e999, 'tolfunhist':0, 'tolfun':0})

	# while not es.stop():
	#     solutions = es.ask()
	#     es.tell(solutions, [score_cma(s) for s in solutions])
	#     es.disp()
	# es.result_pretty()



	with EvalParallel2(score_cma, es.popsize + 1) as eval_all:
		while not es.stop():
			X = es.ask()
			es.tell(X, eval_all(X))
			es.disp()
	es.result_pretty()
	import pdb
	pdb.set_trace()
