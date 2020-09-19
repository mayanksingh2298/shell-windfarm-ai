#andha crude local search
import numpy as np
from evaluate import checkConstraints, binWindResourceData, getAEP, loadPowerCurve, getTurbLoc, preProcessing
from datetime import datetime
import random
from args import make_args
from tqdm import tqdm
from utils import score, initialise_valid, initialise_periphery, min_dist_from_rest
from constants import *

args = make_args()

EPSILON = args.step
RANDOM_RESTART_THRESH = 1000000000
# RANDOM_EPS = args.random_eps
RANDOM_EPS = True
# DIRECTIONS = args.directions
DIRECTIONS = 36
IMPROVEMENT_THRESH = 0.0001
GREEDY_TURBINE_PROB = 0.5

if RANDOM_EPS:
	# RANDOM_RESTART_THRESH = 10*RANDOM_RESTART_THRESH
	# LOWER_LIM = 50
	# UPPER_LIM = 500
	LOWER_LIM = 1
	UPPER_LIM = 500

def save_csv(coords):
	# f = open("submissions/temp_{}_avg_aep_{}m_jump_{}_directions_{}_iterations_{}.csv"
		# .format(score(coords, wind_inst_freq), EPSILON,DIRECTIONS,iteration, str(datetime.now()).replace(':','')), "w")
	f = open("submissions/local_search_GA.csv",'w')
	np.savetxt(f, coords, delimiter=',', header='x,y', comments='', fmt='%1.8f')
	f.close()




if __name__ == "__main__":
	
	#initialise coords with working values
	years = [2007, 2008, 2009, 2013, 2014, 2015, 2017]
	# years = [2007]
	year_wise_dist = np.array([binWindResourceData('../../data/WindData/wind_data_{}.csv'.format(year)) for year in years])
	wind_inst_freq = np.mean(year_wise_dist, axis = 0) #over all years
	# coords   =  getTurbLoc('../data/turbine_loc_test.csv') #supposed to be a numpy array of shape 50 x 2
	#to check initialiser

	iteration = -1


	# coords = initialise_periphery()
	# coords = initialise_valid()
	coords = getTurbLoc("submissions/best_yet-528.95.csv")

	DELTA = (2*np.pi)/DIRECTIONS

	total_iterations = 0
	num_restarts = 0
	iters_with_no_inc = 0
	# while(True):
	for ii in tqdm(range(9999999999999)):
		if iters_with_no_inc >= RANDOM_RESTART_THRESH:
			save_csv(coords)
			iters_with_no_inc = 0
			iteration = 0
			num_restarts += 1
			coords = initialise_valid()

		if RANDOM_EPS:
			EPSILON = np.random.uniform(LOWER_LIM, UPPER_LIM)

		total_iterations += 1
		iteration += 1

		old_score,turbine_powers = score(coords,wind_inst_freq, False, get_each_turbine_power=True) 
		#select turbine to move
		if(random.random()<=GREEDY_TURBINE_PROB):
			chosen = np.argmin(turbine_powers)
		else:
			chosen = np.random.randint(0,50) #50 is not included
		

		#now lets see whether we can improve upon our score
		# coords = coords.copy()
		x, y = coords[chosen]



		best_coords = None
		best_improvement = 0

		######################################################################################################################
		# copied = coords.copy() # an undo can be done to optimise later
		# def random_change(n):
		# 	global copied
		# 	# makes changes in n turbines randomly (with greedy choosing with greedy probability)
		# 	for i in range(n):
		# 		if(random.random()<=GREEDY_TURBINE_PROB):
		# 			chosen = np.argmin(turbine_powers)
		# 		else:
		# 			chosen = np.random.randint(0,50) #50 is not included
		# 		option = random.randint(0,DIRECTIONS-1)
		# 		angle = option*DELTA
		# 		new_x = copied[chosen][0] + EPSILON*np.cos(angle)
		# 		new_y = copied[chosen][1] + EPSILON*np.sin(angle)
		# 		min_d = min_dist_from_rest(chosen, copied, new_x, new_y)
		# 		if not (50 < new_x < 3950 and 50 < new_y < 3950):
		# 			copied = coords.copy()
		# 			# print("problem",min_d)
		# 			return
		# 		if min_d <= 400 + PRECISION:
		# 			copied = coords.copy()
		# 			# print("problem",min_d)
		# 			return 
		# 		copied[chosen][0], copied[chosen][1] = new_x, new_y 
		# random_change(3) # NEW CORDS ARE PRESENT IN COPIED
		# new_score = score(copied, wind_inst_freq)
		# improvement = new_score - old_score
		# # print(improvement)
		# if improvement >= IMPROVEMENT_THRESH:
		# 	best_improvement = improvement
		# 	best_coords = copied
		# 	coords = best_coords
		# 	print("new_score:",new_score,"Iteration:",total_iterations)
		# 	save_csv(coords)
		# 	iters_with_no_inc = 0 
		# else:
		# 	iters_with_no_inc += 1
		######################################################################################################################


		for option in range(DIRECTIONS):
			angle = option*DELTA

			new_x = x + EPSILON*np.cos(angle)
			new_y = y + EPSILON*np.sin(angle)


			#check coords yahi pe
			if not (50 < new_x < 3950 and 50 < new_y < 3950):
				continue
				#just need to check the latest point
			#also check dist from nbrs of this new pt
			min_d = min_dist_from_rest(chosen, coords, new_x, new_y)
			if min_d <= 400 + PRECISION:
				continue

			copied = coords.copy() # an undo can be done to optimise later
			copied[chosen][0], copied[chosen][1] = new_x, new_y 
			new_score = score(copied, wind_inst_freq)
			improvement = new_score - old_score

			# if improvement >= best_improvement:
			if improvement >= IMPROVEMENT_THRESH:
				best_improvement = improvement
				best_coords = copied
				coords = best_coords
				print("new_score:",new_score,"Iteration:",total_iterations)
				save_csv(coords)
				iters_with_no_inc = 0 
				break
		else:
			iters_with_no_inc += 1

