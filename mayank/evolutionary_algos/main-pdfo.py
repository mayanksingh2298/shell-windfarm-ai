from pdfo import pdfo, Bounds, LinearConstraint, NonlinearConstraint
# If SciPy (version 1.1 or above) is installed, then Bounds, LinearConstraint,
# and NonlinearConstraint can alternatively be imported from scipy.optimize.
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
from utils import score, score_pdfo, initialise_valid, initialise_periphery, min_dist_from_rest, delta_score, delta_check_constraints, fetch_movable_segments
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


if __name__ == '__main__':
	# if args.year is None:
	# 	years = YEARS
	# else:
	# 	years = [args.year]
	# year_wise_dist = np.array([binWindResourceData('../../data/WindData/wind_data_{}.csv'.format(year)) for year in years])
	# wind_inst_freq = np.mean(year_wise_dist, axis = 0) #over all years


	#define the starting point [x1,y1,x2,y2,...,x50,y50]
	x0 = getTurbLoc(sys.argv[1]).reshape(-1)

	#define options
	options={'honour_x0':True, 'rhobeg':2, 'rhoend': 1e-5000}


	#define the bounds
	lb = [50] * 100
	ub = [3950] * 100
	bounds = Bounds(lb,ub)

	#start it
	print("Started pdfo")
	res = pdfo(score_pdfo, x0, bounds=bounds, options=options)
	print(res)