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
RANDOM_RESTART_THRESH = 100
RANDOM_EPS = args.random_eps
DIRECTIONS = args.directions

if RANDOM_EPS:
	RANDOM_RESTART_THRESH = 500*RANDOM_RESTART_THRESH
	LOWER_LIM = 0.1
	upper_lims = [3, 20, 200, 600, 5000]
	lim_sizes = ["teeny tiny", "super small", "small", "medium", "large"]

def save_csv(coords):
	f = open("submissions/temp_{}_avg_aep_{}m_jump_{}_directions_{}_iterations_{}.csv"
		.format(score(coords, wind_inst_freq), EPSILON,DIRECTIONS,iteration, str(datetime.now()).replace(':','')), "w")
	np.savetxt(f, coords, delimiter=',', header='x,y', comments='', fmt='%1.8f')
	f.close()


# def debug(coords):
# 	for i in range(50):
# 		for j in range(i+1,50):
# 			x1, y1 = coords[i]
# 			x2, y2 = coords[j]
# 			if (x1-x2)**2 + (y1-y2)**2 <= 400*400:
# 				print("error")
# 				print(i,j)
# 				print(coords[i], coords[j])

if __name__ == "__main__":
	
	#initialise coords with working values
	years = [2007, 2008, 2009, 2013, 2014, 2015, 2017]
	# years = [2007]
	year_wise_dist = np.array([binWindResourceData('../data/WindData/wind_data_{}.csv'.format(year)) for year in years])
	wind_inst_freq = np.mean(year_wise_dist, axis = 0) #over all years
	# coords   =  getTurbLoc('../data/turbine_loc_test.csv') #supposed to be a numpy array of shape 50 x 2
	#to check initialiser

	iteration = 0


	# print("random model")
	# last = 500
	# for _ in tqdm(range(10000000000)):
	# 	coords = initialise_valid()
	# 	# coords = initialise_periphery()
	# 	# if not checkConstraints(coords, turb_diam):
	# 	# 	print("wrong initialiser")
	# 	# 	break
	# 	s = score(coords)
	# 	if s == MINIMUM:
	# 		print("ERROR")
	# 	if s > last:
	# 		last = s
	# 		save_csv(coords)
	# 		print("found {}".format(s))

	# sys.exit()

	# coords = initialise_periphery()
	coords = initialise_valid()

	DELTA = (2*np.pi)/DIRECTIONS

	total_iterations = 0
	num_restarts = 0
	iters_with_no_inc = 0
	while(True):
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
			sample = np.random.uniform(0,1)
			lim_id = int(sample*len(upper_lims)) 
			# if  < 0.7:
			EPSILON = np.random.uniform(LOWER_LIM, upper_lims[lim_id])
			print("considering a {} step - {}".format(lim_sizes[lim_id], EPSILON))

		total_iterations += 1
		iteration += 1
		chosen = np.random.randint(0,50) #50 is not included
		#now lets see whether we can improve upon our score

		# coords = coords.copy()

		#considering just
		best_coords = None
		best_improvement = 0
		best_windmill = None
		old_score, original_deficit = score(coords,wind_inst_freq, True, True) 
		print("current average : {}".format(old_score))
		for chosen in range(50):
			x, y = coords[chosen]
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
				# new_score = score(copied, wind_inst_freq)
				new_score, _ = delta_score(coords, wind_inst_freq, chosen, new_x, new_y, original_deficit)

				improvement = new_score - old_score

				if improvement >= best_improvement:
					best_improvement = improvement
					best_coords = copied
					best_windmill = chosen
			

		if best_coords is None:
			print("no improvement in any direction; happened {} consecutive times before this".format( iters_with_no_inc))
			iters_with_no_inc += 1
		else:
			print("Chose windmill {} and got an improvement of {} units in the average AEP".format(best_windmill, best_improvement))
			iters_with_no_inc = 0 #because we are considering such consecutive iters	
			# score(chosen, )
			coords = best_coords

		if iteration%5000 == 0:
			print("saving")
			save_csv(coords)

	print("DONE")		 
	save_csv(coords)