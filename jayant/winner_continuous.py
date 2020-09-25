#smartly choosing where to jump to
import numpy as np
from evaluate import checkConstraints, binWindResourceData, getAEP, loadPowerCurve, getTurbLoc, preProcessing
from datetime import datetime
import random
import time
from args import make_args
from tqdm import tqdm
from utils import score, initialise_valid, initialise_periphery, min_dist_from_rest, delta_score, delta_check_constraints, fetch_movable_segments
from utils import min_dis,  initialise_file
from constants import *


args = make_args()
NN = 1
GREEDY = 0.7

def save_csv(coords):
	f = open("submissions/{}temp_{}_avg_aep_{}_iterations_{}.csv"
		.format("" if args.year is None else "specific_year_{}".format(args.year), score(coords, wind_inst_freq),iteration, str(datetime.now()).replace(':','')), "w")
	np.savetxt(f, coords, delimiter=',', header='x,y', comments='', fmt='%1.8f')
	f.close()

if __name__ == "__main__":

	if args.year is None:
		years = [2007, 2008, 2009, 2013, 2014, 2015, 2017]
	else:
		years = [args.year]
	year_wise_dist = np.array([binWindResourceData('../data/WindData/wind_data_{}.csv'.format(year)) for year in years])
	wind_inst_freq = np.mean(year_wise_dist, axis = 0) #over all years

	iteration = 0
	if args.file is None:
		coords = initialise_periphery()
	else:
		coords = initialise_file(args.file)

	total_iterations = 0
	num_restarts = 0
	iters_with_no_inc = 0
	old_score, original_deficit = score(coords,wind_inst_freq, True, True, True) 

	# sys.exit()
	while(True):
		if iteration%5000 == 0:
			score(coords,wind_inst_freq,True)
			time.sleep(5)

		if iteration%50000 == 0:
			print("saving")
			save_csv(coords)

		total_iterations += 1
		iteration += 1
		chosen = np.random.randint(0,50) #50 is not included
		x, y = coords[chosen]
		found = False
		best_ind = -1
		best_score = old_score

		#code here
		# sample a direction
		# get all possible segments
		# use the midpoints 

		# direction = np.random.uniform(0, 2*np.pi)
		if np.random.uniform() > GREEDY:
			direction = np.random.uniform(0, 2*np.pi)
		else:
			x, y = coords[chosen]
			vi = np.array([x,y])
			dists = []
			points = []
			for i, (x_dash, y_dash) in enumerate(coords):
			    if i != chosen:
			       dists.append(((x - x_dash)**2 + (y - y_dash)**2))
			       points.append((x_dash, y_dash))
			# dists.sort()
			indices = np.argpartition(dists, NN)

			# v = 1e-5*np.ones(2)
			v = np.zeros(2)
			for i in indices[:NN]:
			    v += np.array([x - points[i][0], y - points[i][1]])
			norm = np.linalg.norm(v)
			if norm < 1e-10:
				direction = np.random.uniform(0, 2*np.pi)
			else:		
				v = v / np.linalg.norm(v)
				theta_v = np.arccos(v[0])
				# direction = np.random.normal(theta_v, 0.1)
				direction = np.random.normal(theta_v, np.pi/12)
				# direction = np.random.normal(theta_v, np.pi/6)



		segments = fetch_movable_segments(coords, chosen, direction)
		
		possibilities = []

		#uniform samples from each seg
		samples = np.random.uniform(size = len(segments))
		possibilities += [((samples[i]*a[0]+ (1-samples[i])*b[0]), (samples[i]*a[1] + (1-samples[i])*b[1])) for i,(a,b) in enumerate(segments)]

		# centres
		possibilities += [((a[0]+ b[0])/2, (a[1] + b[1])/2) for a,b in segments]
		# #lefts
		possibilities += [((0.999*a[0]+ 0.001*b[0]), (0.999*a[1] + 0.001*b[1])) for a,b in segments]
		# # #rights
		possibilities += [((0.001*a[0]+ 0.999*b[0]), (0.001*a[1] + 0.999*b[1])) for a,b in segments]
		

		for ind, (new_x, new_y) in enumerate(possibilities):
			if not delta_check_constraints(coords, chosen, new_x, new_y):
				print("ERROR")
				# sys.exit()
				continue

			# new_score = score(copied, wind_inst_freq)
			new_score, new_deficit = delta_score(coords, wind_inst_freq, chosen, new_x, new_y, original_deficit, continuous = True)
			# improvement = new_score - old_score

			if new_score >= best_score:
				best_score = new_score
				best_ind = ind
				best_deficit = new_deficit


		if best_ind == -1:
			# print("Chose windmill {} but no improvement in this direction; happened {} consecutive times before this".format(chosen, iters_with_no_inc))
			iters_with_no_inc += 1

		else:
			print("Total iter num: {} ".format(total_iterations))
			print("Chose windmill {} and got an improvement of {} units in the average AEP".format(chosen, best_score - old_score))
			iters_with_no_inc = 0 #because we are considering such consecutive iters	
			# score(chosen, )
			coords[chosen][0], coords[chosen][1] = possibilities[best_ind]
			old_score = best_score
			original_deficit = best_deficit
			print("approximate average : {}".format(old_score))
			print()

	print("DONE")		 
	save_csv(coords)
