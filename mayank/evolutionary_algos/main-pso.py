
# Import modules
import numpy as np

# Import PySwarms
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
import numpy as np
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
from utils import score_pso, initialise_valid, initialise_periphery, min_dist_from_rest, delta_score, delta_check_constraints, fetch_movable_segments
from utils import min_dis, initialise_file
from constants import *
from datetime import datetime


WINNER_CSV = "data/winner.csv"
CHILDREN_DATA_ROOT = "data/children"

def save_csv(coords,path):
	f = open(path,"w")
	np.savetxt(f, coords, delimiter=',', header='x,y', comments='', fmt='%1.8f')
	f.close()


if __name__ == '__main__':
	NPARTICLES = 100
	#define the starting point [x1,y1,x2,y2,...,x50,y50]
	# x0 = getTurbLoc(sys.argv[1]).reshape(-1)
	# sigma0 = 0.01
	x0 = getTurbLoc(sys.argv[1]).reshape(-1)

	# #define options
	options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}


	# #define the bounds
	max_bound = 3950 * np.ones(100)
	min_bound = 50 * np.ones(100)
	bounds = (min_bound, max_bound)

	# #start it
	print("Started PSO")
	optimizer = ps.single.GlobalBestPSO(n_particles=NPARTICLES, dimensions=100, options=options, bounds=bounds, init_pos=np.array([x0]*NPARTICLES))
	cost, pos = optimizer.optimize(score_pso, n_processes=10, iters=1000)
	import pdb
	pdb.set_trace()






	# Set-up hyperparameters

	# Call instance of PSO

	# Perform optimization