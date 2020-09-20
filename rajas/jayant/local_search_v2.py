#best first 
import numpy as np
from evaluate import checkConstraints, binWindResourceData, getAEP, loadPowerCurve, getTurbLoc, preProcessing
from datetime import datetime
import random
from args import make_args
from tqdm import tqdm
from utils import score, initialise_valid, initialise_periphery, min_dist_from_rest, delta_score
from constants import *

args = make_args()

EPSILON = args.step
RANDOM_RESTART_THRESH = 100* 500
RANDOM_EPS = True
DIRECTIONS = args.directions
RANDOM_NEG = True
weighting = None

def to_radian(x):
	return (x * np.pi)/180.0

def save_csv(coords):
	f = open("submissions/temp_{}_avg_aep_{}m_jump_{}_directions_{}_iterations_{}.csv"
		.format(score(coords, wind_inst_freq), EPSILON,DIRECTIONS,iteration, str(datetime.now()).replace(':','')), "w")
	np.savetxt(f, coords, delimiter=',', header='x,y', comments='', fmt='%1.8f')
	f.close()

if __name__ == "__main__":
	
	years = [2007, 2008, 2009, 2013, 2014, 2015, 2017]
	year_wise_dist = np.array([binWindResourceData('../../data/WindData/wind_data_{}.csv'.format(year)) for year in years])
	wind_inst_freq = np.mean(year_wise_dist, axis = 0) #over all years
	if weighting is not None:
		prob_dist = wind_inst_freq.reshape(n_slices_drct, -1)
		if "dir" in weighting:
			prob_dist = prob_dist.sum(axis = 1)
		else:
			prob_dist = np.matmul(prob_dist, np.arange(2,32,2))

	iteration = 0

	coords = initialise_valid()

	DELTA = (2*np.pi)/DIRECTIONS

	total_iterations = 0
	num_restarts = 0
	iters_with_no_inc = 0
	do = False
	while(True):
		if iteration%5000 == 0:
			print("saving")
			save_csv(coords)
		print()
		print("Total iter num: {}, num restarts: {}, local iteration number {}".format(total_iterations, num_restarts, iteration))
		
		if iters_with_no_inc >= RANDOM_RESTART_THRESH:
			save_csv(coords)
			iters_with_no_inc = 0
			iteration = 0
			num_restarts += 1
			coords = initialise_valid()
			continue

		if RANDOM_EPS:
			LOWER_LIM = 10
			# upper_lims = [500]
			# upper_lim = random.choice(upper_lims)
			upper_lim = 500
			EPSILON = np.random.uniform(LOWER_LIM, upper_lim)
			print("considering a step - {}".format(EPSILON))

		total_iterations += 1
		iteration += 1
		chosen = np.random.randint(0,50)

		best_coords = None
		best_improvement = 0
		best_windmill = None
		old_score, original_deficit = score(coords,wind_inst_freq, True, True)
		print("current average : {}".format(old_score))
		if weighting is None:
			angle_options = np.arange(0, 2*np.pi, DELTA)
		else:
			wind_angle = np.random.choice(np.arange(0, 360, 10), p=prob_dist)
			angle_options = []
			for i in range(4):
				angle_options.append(to_radian((wind_angle + 90 * i) % 360))

		for chosen in range(50):
			x, y = coords[chosen]
			for angle in angle_options:
				new_x = x + EPSILON*np.cos(angle)
				new_y = y + EPSILON*np.sin(angle)

				if not (50 < new_x < 3950 and 50 < new_y < 3950):
					continue

				min_d = min_dist_from_rest(chosen, coords, new_x, new_y)
				if min_d <= 400 + PRECISION:
					continue

				new_score, _ = delta_score(coords, wind_inst_freq, chosen, new_x, new_y, original_deficit)

				improvement = new_score - old_score

				if improvement >= best_improvement:
					copied = coords.copy()
					copied[chosen][0], copied[chosen][1] = new_x, new_y
					best_improvement = improvement
					best_coords = copied
					best_windmill = chosen
			

		if best_coords is None:
			print("no improvement in any direction; happened {} consecutive times before this".format( iters_with_no_inc))
			iters_with_no_inc += 1
			if RANDOM_NEG:
				if iters_with_no_inc%10 == 0 or do:
					chosen = np.random.randint(0,50)
					x, y = coords[chosen]
					option = random.choice(range(DIRECTIONS))
					angle = option*DELTA
					EPSILON = np.random.uniform(1, 20)
					new_x = x + EPSILON*np.cos(angle)
					new_y = y + EPSILON*np.sin(angle)
					if not (50 < new_x < 3950 and 50 < new_y < 3950):
						do = True
						continue

					min_d = min_dist_from_rest(chosen, coords, new_x, new_y)
					if min_d <= 400 + PRECISION:
						do = True
						continue
					coords[chosen][0], coords[chosen][1] = new_x, new_y
					do = False
					print("took a random {} step in {} direction for windmill {}".format(EPSILON, option, chosen))
		else:
			print("Chose windmill {} and got an improvement of {} units in the average AEP".format(best_windmill, best_improvement))
			iters_with_no_inc = 0
			coords = best_coords


	print("DONE")		 
	save_csv(coords)