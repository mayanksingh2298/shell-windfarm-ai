#smartly choosing where to jump to
import numpy as np
from evaluate import checkConstraints, binWindResourceData, getAEP, loadPowerCurve, getTurbLoc, preProcessing
from datetime import datetime
import random
from args import make_args
from tqdm import tqdm
from utils import score, initialise_valid, initialise_periphery, min_dist_from_rest, delta_score, delta_check_constraints, fetch_movable_segments
from utils import min_dis
from constants import *

args = make_args()
NN = 2
GREEDY = 0.5

def save_csv(coords):
	f = open("submissions/temp_{}_avg_aep_{}_iterations_{}.csv"
		.format(score(coords, wind_inst_freq),iteration, str(datetime.now()).replace(':','')), "w")
	np.savetxt(f, coords, delimiter=',', header='x,y', comments='', fmt='%1.8f')
	f.close()

if __name__ == "__main__":

	years = [2007, 2008, 2009, 2013, 2014, 2015, 2017]
	year_wise_dist = np.array([binWindResourceData('../data/WindData/wind_data_{}.csv'.format(year)) for year in years])
	wind_inst_freq = np.mean(year_wise_dist, axis = 0) #over all years

	iteration = -1
	coords = initialise_periphery()

	total_iterations = 0
	num_restarts = 0
	iters_with_no_inc = 0
	old_score, original_deficit = score(coords,wind_inst_freq, True, True, False) 

	# sys.exit()
	while(True):
		if iteration%50000 == 0:
			print("saving")
			save_csv(coords)

		total_iterations += 1
		iteration += 1
		# chosen = np.random.randint(0,50) #50 is not included
		# x, y = coords[chosen]
		found = False
		best_chosen = -1
		best_score = old_score

		#looking for a movable point, small changes not allowed basically
		before_finding = 0
		while(not found):
			new_x = np.random.uniform(50,3950)
			new_y = np.random.uniform(50,3950)
			before_finding += 1
			#chosen val = -1 means we check against everyone already there
			if delta_check_constraints(coords, -1, new_x, new_y):
				found = True

		# print("time wasted finding : {} many samples ".format(before_finding))

		for chosen in range(50):
			if not delta_check_constraints(coords, chosen, new_x, new_y):
				print("ERROR")
				sys.exit()
				# continue

			# new_score = score(copied, wind_inst_freq)
			new_score, new_deficit = delta_score(coords, wind_inst_freq, chosen, new_x, new_y, original_deficit)
			# improvement = new_score - old_score

			if new_score >= best_score:
				best_score = new_score
				best_chosen = chosen
				best_deficit = new_deficit


		if best_chosen == -1:
			# print("Chose windmill {} but no improvement in this direction; happened {} consecutive times before this".format(chosen, iters_with_no_inc))
			iters_with_no_inc += 1

		else:
			print("Chose windmill {} and got an improvement of {} units in the average AEP".format(best_chosen, best_score - old_score))
			print("Total iter num: {} ".format(total_iterations))
			iters_with_no_inc = 0 #because we are considering such consecutive iters	
			# score(chosen, )
			coords[best_chosen][0], coords[best_chosen][1] = new_x, new_y
			old_score = best_score
			original_deficit = best_deficit
			print("average : {}".format(old_score))
			print()

	print("DONE")		 
	save_csv(coords)
