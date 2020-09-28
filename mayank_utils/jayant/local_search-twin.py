#smartly choosing where to jump to
import numpy as np
from evaluate import checkConstraints, binWindResourceData, getAEP, loadPowerCurve, getTurbLoc, preProcessing
from datetime import datetime
import random
import time
from args import make_args
from tqdm import tqdm
from utils import score, initialise_valid, initialise_periphery, initialise_periphery_mayank, min_dist_from_rest, delta_score2, delta_check_constraints2, fetch_movable_segments
from utils import min_dis
from constants import *

# args = make_args()
# NN = 2
# GREEDY = 0.5
IMPROVE_THRESH = 0.001
def save_csv(coords):
	f = open("submissions/local_search_twin.csv",'w')
	np.savetxt(f, coords, delimiter=',', header='x,y', comments='', fmt='%1.8f')
	f.close()

if __name__ == "__main__":

	years = [2007, 2008, 2009, 2013, 2014, 2015, 2017]
	year_wise_dist = np.array([binWindResourceData('../../data/WindData/wind_data_{}.csv'.format(year)) for year in years])
	wind_inst_freq = np.mean(year_wise_dist, axis = 0) #over all years

	iteration = -1
	# coords = initialise_valid()
	# coords = initialise_periphery_mayank()
	coords = getTurbLoc("submissions/local_search_2.csv")

	total_iterations = 0
	num_restarts = 0
	iters_with_no_inc = 0
	old_score, original_deficit = score(coords,wind_inst_freq, True, True) 
	for ii in tqdm(range(9999999999999)):
		total_iterations += 1
		iteration += 1

		#one other idea is to only choose from the last 14 points because they are the ones not on perimeter
		chosen1 = np.random.randint(36,50) #50 is not included
		chosen2 = np.random.randint(36,50) #50 is not included



		x1, y1 = coords[chosen1]
		x2, y2 = coords[chosen2]

		found = False
		best_ind1 = -1
		best_ind2 = -1

		best_score = old_score

		# for now choosing the perpendicular direction

		direction = np.random.uniform(0, 2*np.pi)
		direction2 = direction + np.pi/2
		segments1 = fetch_movable_segments(coords, chosen1, direction)
		segments2 = fetch_movable_segments(coords, chosen2, direction2)

		
		possibilities1 = []
		possibilities2 = []


		#uniform samples from each seg
		samples = np.random.uniform(size = len(segments1))
		possibilities1 += [((samples[i]*a[0]+ (1-samples[i])*b[0]), (samples[i]*a[1] + (1-samples[i])*b[1])) for i,(a,b) in enumerate(segments1)]
		#centres
		possibilities1 += [((a[0]+ b[0])/2, (a[1] + b[1])/2) for a,b in segments1]
		# #lefts
		possibilities1 += [((0.999*a[0]+ 0.001*b[0]), (0.999*a[1] + 0.001*b[1])) for a,b in segments1]
		# # #rights
		possibilities1 += [((0.001*a[0]+ 0.999*b[0]), (0.001*a[1] + 0.999*b[1])) for a,b in segments1]

		#uniform samples from each seg
		samples = np.random.uniform(size = len(segments2))
		possibilities2 += [((samples[i]*a[0]+ (1-samples[i])*b[0]), (samples[i]*a[1] + (1-samples[i])*b[1])) for i,(a,b) in enumerate(segments2)]
		#centres
		possibilities2 += [((a[0]+ b[0])/2, (a[1] + b[1])/2) for a,b in segments2]
		# #lefts
		possibilities2 += [((0.999*a[0]+ 0.001*b[0]), (0.999*a[1] + 0.001*b[1])) for a,b in segments2]
		# # #rights
		possibilities2 += [((0.001*a[0]+ 0.999*b[0]), (0.001*a[1] + 0.999*b[1])) for a,b in segments2]
		

		constraitsFalse = 0
		for ind1, (new_x1, new_y1) in enumerate(possibilities1):
			for ind2, (new_x2, new_y2) in enumerate(possibilities2):
				# new_coords = coords.copy()
				# new_coords[chosen1][0] = new_x1
				# new_coords[chosen1][1] = new_y1
				# new_coords[chosen2][0] = new_x2
				# new_coords[chosen2][1] = new_y2

				# if not delta_check_constraints(coords, chosen, new_x, new_y):
				# 	continue
				if not delta_check_constraints2(coords, chosen1, chosen2, new_x1, new_y1, new_x2, new_y2):
					constraitsFalse+=1
					continue

				# new_score = score(copied, wind_inst_freq)
				new_score, new_deficit = delta_score2(coords, wind_inst_freq, chosen1, chosen2, new_x1, new_y1, new_x2, new_y2, original_deficit)
				# improvement = new_score - old_score
				# print(new_score - best_score)
				# import pdb
				# pdb.set_trace()
				flag = 0
				if new_score >= best_score+IMPROVE_THRESH:
					flag = 1
					best_score = new_score
					best_ind1 = ind1
					best_ind2 = ind2
					best_deficit = new_deficit
					break
			if(flag==1):
				break
		# print("Constraints have been wronged {} times in this iteration!".format(constraitsFalse))
		if best_ind1 == -1:
			iters_with_no_inc += 1

		else:
			try:
				iters_with_no_inc = 0 #because we are considering such consecutive iters	
				# score(chosen, )
				coords[chosen1][0], coords[chosen1][1] = possibilities1[best_ind1]
				coords[chosen2][0], coords[chosen2][1] = possibilities2[best_ind2]
			except Exception as e:
				print(e)
				import pdb
				pdb.set_trace()

			old_score = best_score
			original_deficit = best_deficit
			print("new_score:",old_score,"Iteration:",total_iterations)	
			save_csv(coords)


	print("DONE")		 
	save_csv(coords)