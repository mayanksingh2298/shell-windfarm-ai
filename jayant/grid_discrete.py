import numpy as np
import math
from evaluate import checkConstraints, binWindResourceData, getAEP, loadPowerCurve, getTurbLoc, preProcessing
import random
from datetime import datetime
from utils import score, initialise_valid, initialise_periphery, min_dist_from_rest, delta_score, initialise_file
from constants import *
import sys
import pandas as pd
from tqdm import tqdm

D = 10
N = 4000/D
DEPTH = 1

def save_csv(coords):
	f = open("submissions/temp_{}_avg_aep_{}_iterations_{}.csv"
		.format(score(coords, wind_inst_freq),iterno, str(datetime.now()).replace(':','')), "w")
	np.savetxt(f, coords, delimiter=',', header='x,y', comments='', fmt='%1.8f')
	f.close()

def check_possible(coords, chosen, new_check):
	new_x = new_check[0]
	new_y = new_check[1]

	if not (50 < new_x < 3950 and 50 < new_y < 3950):
		return False

	min_d = min_dist_from_rest(chosen, coords, new_x, new_y)
	if min_d <= 400 + PRECISION:
	   return False
	return True

def get_coords(i, j):
	return np.array([np.random.uniform(i*D, (i+1)*D), np.random.uniform(j*D, (j+1)*D)])

if __name__ == "__main__":
	years = [2007, 2008, 2009, 2013, 2014, 2015, 2017]
	year_wise_dist = np.array([binWindResourceData('../data/WindData/wind_data_{}.csv'.format(year)) for year in years])
	wind_inst_freq = np.mean(year_wise_dist, axis = 0) #over all years

	if len(sys.argv) > 1:
		coords = pd.read_csv(sys.argv[1]).to_numpy()
		coords = (coords / D).astype(int).astype(float) * D + D/2
	else:
		coords = initialise_valid_discrete()

	old_score, original_deficit= score(coords, wind_inst_freq, True, True)
	
	for iterno in tqdm(range(1000000)):
		chosen = np.random.randint(0,50)
		old_i, old_j = int(coords[chosen][0] / D) * D, int(coords[chosen][1] / D) * D
		
		i,j = np.random.randint(0,N), np.random.randint(0,N)
		new_point = get_coords(i,j)
		while not check_possible(coords, chosen, new_point):
			i,j = np.random.randint(0,N), np.random.randint(0,N)
			new_point = get_coords(i,j)
		
		new_score, new_deficit = delta_score(coords, wind_inst_freq, chosen, new_point[0], new_point[1], original_deficit)
		
		if new_score >= old_score:
			old_score = new_score
			original_deficit = new_deficit
			coords[chosen] = new_point
			print (old_score)

		if iterno % 100000 == 0:
			save_csv(coords)

