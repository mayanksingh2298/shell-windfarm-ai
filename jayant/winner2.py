#smartly choosing where to jump to
import numpy as np
from evaluate import checkConstraints, binWindResourceData, getAEP, loadPowerCurve, getTurbLoc, preProcessing
from datetime import datetime
import random
from args import make_args
from tqdm import tqdm
from utils import score, initialise_valid, initialise_periphery, min_dist_from_rest, delta_score, delta_check_constraints, fetch_movable_segments
from constants import *

args = make_args()
NN = 2
GREEDY = 0.7

def save_csv(coords):
	f = open("submissions/temp_{}_avg_aep_{}_iterations_{}.csv"
		.format(score(coords, wind_inst_freq),iteration, str(datetime.now()).replace(':','')), "w")
	np.savetxt(f, coords, delimiter=',', header='x,y', comments='', fmt='%1.8f')
	f.close()

if __name__ == "__main__":

	years = [2007, 2008, 2009, 2013, 2014, 2015, 2017]
	year_wise_dist = np.array([binWindResourceData('../data/WindData/wind_data_{}.csv'.format(year)) for year in years])
	wind_inst_freq = np.mean(year_wise_dist, axis = 0) #over all years

	iteration = 0
	coords = initialise_periphery()

	total_iterations = 0
	num_restarts = 0
	iters_with_no_inc = 0
	old_score, original_deficit = score(coords,wind_inst_freq, True, True) 
	while(True):
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
				# direction = np.random.normal(theta_v, np.pi/6)
				direction = np.random.normal(theta_v, np.pi/6)



		segments = fetch_movable_segments(coords, chosen, direction)
		
		possibilities = []

		#uniform samples from each seg
		# possibilities += [(np.random.uniform(min(a[0],b[0]), max(a[0], b[0])), np.random.uniform(min(a[1],b[1]), max(a[1], b[1]))) for a,b in segments]
		#centres
		# possibilities += [((a[0]+ b[0])/2, (a[1] + b[1])/2) for a,b in segments]
		# #lefts
		# possibilities += [((0.9999*a[0]+ 0.0001*b[0]), (0.9999*a[1] + 0.0001*b[1])) for a,b in segments]
		# #rights
		# possibilities += [((0.0001*a[0]+ 0.9999*b[0]), (0.0001*a[1] + 0.9999*b[1])) for a,b in segments]
		

		for ind, ((a0, a1), (b0,b1)) in enumerate(segments):
			# if not delta_check_constraints(coords, chosen, new_x, new_y):
			# 	print("ERROR")
			# 	sys.exit()
			# 	continue
			# new_x = (a0 + b0)/2
			# new_y = (a1 + b1)/2
			new_x= np.random.uniform(min(a0,b0), max(a0, b0))
			new_y= np.random.uniform(min(a1,b1), max(a1, b1))
			# new_score = score(copied, wind_inst_freq)
			new_score, new_deficit = delta_score(coords, wind_inst_freq, chosen, new_x, new_y, original_deficit)
			# improvement = new_score - old_score

			if new_score >= old_score:
				coords[chosen][0], coords[chosen][1] = new_x, new_y
				original_deficit = new_deficit
				print()	
				print("Total iter num: {} ".format(total_iterations))
				print("Chose windmill {} and got an improvement of {} units in the average AEP".format(chosen, new_score - old_score))
				old_score = new_score
				print("average : {}".format(old_score))

	print("DONE")		 
	save_csv(coords)
