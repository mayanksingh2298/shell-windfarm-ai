#smartly choosing where to jump to
import numpy as np
from evaluate import checkConstraints, binWindResourceData, getAEP, loadPowerCurve, getTurbLoc, preProcessing
from datetime import datetime
import random
from args import make_args
from tqdm import tqdm
from utils import score, initialise_valid, initialise_periphery,initialise_periphery_mayank, min_dist_from_rest, delta_score, delta_check_constraints, fetch_movable_segments
from utils import min_dis
from constants import *

# args = make_args()
# NN = 2
# GREEDY = 0.5
IMPROVE_THRESH = 0.001
MAP_CHANGE_THRESH = 10000

def get_random_wind_frequency(p=0.5):
	while True:
		a1 = random.random()<=p
		a2 = random.random()<=p
		a3 = random.random()<=p
		a4 = random.random()<=p
		a5 = random.random()<=p
		a6 = random.random()<=p
		a7 = random.random()<=p
		if(a1 or a2 or a3 or a4 or a5 or a6 or a7):
			break
	years = []
	if(a1):
		years.append(2007)
	if(a2):
		years.append(2008)
	if(a3):
		years.append(2009)
	if(a4):
		years.append(2013)
	if(a5):
		years.append(2014)
	if(a6):
		years.append(2015)
	if(a7):
		years.append(2017)
	year_wise_dist = np.array([binWindResourceData('../../data/WindData/wind_data_{}.csv'.format(year)) for year in years])
	wind_inst_freq = np.mean(year_wise_dist, axis = 0)
	return wind_inst_freq,years

def save_csv(coords):
	f = open("submissions/rajas_38init.csv",'w')
	np.savetxt(f, coords, delimiter=',', header='x,y', comments='', fmt='%1.8f')
	f.close()

if __name__ == "__main__":

	years = [2007, 2008, 2009, 2013, 2014, 2015, 2017]
	# years = [2015, 2017]
	year_wise_dist = np.array([binWindResourceData('../../data/WindData/wind_data_{}.csv'.format(year)) for year in years])
	wind_inst_freq_og = np.mean(year_wise_dist, axis = 0) #over all years

	iteration = 0
	coords = initialise_periphery_mayank()
	# coords = getTurbLoc("submissions/local_search_2.csv")

	total_iterations = 0
	num_restarts = 0
	iters_with_no_inc = 0

	wind_inst_freq,_ = get_random_wind_frequency()
	old_score, original_deficit = score(coords,wind_inst_freq, True, True) 
	old_score_valid, original_deficit_valid = score(coords,wind_inst_freq_og, True, True) 
	

	for ii in tqdm(range(9999999999999)):
		if(iters_with_no_inc%MAP_CHANGE_THRESH==0 and iters_with_no_inc!=0):
			wind_inst_freq,new_years = get_random_wind_frequency()
			old_score, original_deficit = score(coords,wind_inst_freq, True, True)
			print("Map changed to {}".format(new_years))
			iters_with_no_inc=0
		total_iterations += 1
		iteration += 1
		chosen = np.random.randint(0,50) #50 is not included
		x, y = coords[chosen]
		found = False
		best_ind = -1
		best_score = old_score
		best_score_valid = old_score_valid


		direction = np.random.uniform(0, 2*np.pi)
		segments = fetch_movable_segments(coords, chosen, direction)
		
		possibilities = []

		#uniform samples from each seg
		samples = np.random.uniform(size = len(segments))
		possibilities += [((samples[i]*a[0]+ (1-samples[i])*b[0]), (samples[i]*a[1] + (1-samples[i])*b[1])) for i,(a,b) in enumerate(segments)]
		#centres
		possibilities += [((a[0]+ b[0])/2, (a[1] + b[1])/2) for a,b in segments]
		# #lefts
		possibilities += [((0.999*a[0]+ 0.001*b[0]), (0.999*a[1] + 0.001*b[1])) for a,b in segments]
		# # #rights
		possibilities += [((0.001*a[0]+ 0.999*b[0]), (0.001*a[1] + 0.999*b[1])) for a,b in segments]
		

		for ind, (new_x, new_y) in enumerate(possibilities):
			if not delta_check_constraints(coords, chosen, new_x, new_y):
				continue

			# new_score = score(copied, wind_inst_freq)
			new_score, new_deficit = delta_score(coords, wind_inst_freq, chosen, new_x, new_y, original_deficit)

			# improvement = new_score - old_score

			if new_score >= best_score+IMPROVE_THRESH:
				best_score = new_score
				best_ind = ind
				best_deficit = new_deficit


		if best_ind == -1:
			iters_with_no_inc += 1

		else:
			new_score_valid, new_deficit_valid = delta_score(coords, wind_inst_freq_og, chosen, new_x, new_y, original_deficit_valid)
			if new_score_valid >= best_score_valid+IMPROVE_THRESH:
				best_score_valid = new_score_valid
				best_deficit_valid = new_deficit_valid

				iters_with_no_inc = 0 #because we are considering such consecutive iters	
				# score(chosen, )
				coords[chosen][0], coords[chosen][1] = possibilities[best_ind]
				old_score = best_score
				original_deficit = best_deficit
				old_score_valid = best_score_valid
				original_deficit_valid = best_deficit_valid
				print("new_score:",old_score,"new_score_valid:",old_score_valid,"Iteration:",total_iterations)	
				save_csv(coords)
			else:
				iters_with_no_inc+=1


	print("DONE")		 
	save_csv(coords)
