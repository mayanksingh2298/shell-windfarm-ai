
import numpy as np
from evaluate import checkConstraints, binWindResourceData, getAEP, loadPowerCurve, getTurbLoc, preProcessing
from datetime import datetime
import random
from args import make_args
from tqdm import tqdm, trange
from utils import score, initialise_valid, initialise_periphery, min_dist_from_rest, delta_score, delta_check_constraints, fetch_movable_segments
from utils import min_dis, initialise_file
from constants import *
import math
import matplotlib.pyplot as plt

args = make_args()
NN = 1
MIN_THRESH = 0
# GREEDY = 0.7

def plot(coords):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_aspect('equal', 'box')
    x = coords[:,0]
    y = coords[:,1]
    ax.scatter(x, y, c='blue')
    plt.show()

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def save_csv(coords):
	f = open("submissions/{}temp_{}_avg_aep_{}_iterations_{}.csv"
		.format("" if args.year is None else "specific_year_{}".format(args.year), round(score(coords, wind_inst_freq),6),iteration, str(datetime.now()).replace(':','')), "w")
	np.savetxt(f, coords, delimiter=',', header='x,y', comments='', fmt='%1.8f')
	f.close()
def save_csv_path(coords,path):
	f = open(path, "w")
	np.savetxt(f, coords, delimiter=',', header='x,y', comments='', fmt='%1.8f')
	f.close()

if __name__ == "__main__":

	min_thresh_to_move = MIN_THRESH
	if args.year is None:
		years = [2007, 2008, 2009, 2013, 2014, 2015, 2017]
	else:
		years = [args.year]
	year_wise_dist = np.array([binWindResourceData('../../data/WindData/wind_data_{}.csv'.format(year)) for year in years])
	wind_inst_freq = np.mean(year_wise_dist, axis = 0) #over all years

	iteration = -1

	if args.file is None:
		coords = initialise_periphery()
	else:
		coords = initialise_file(args.file)

	total_iterations = 0
	num_restarts = 0
	iters_with_no_inc = 0
	old_score, original_deficit = score(coords,wind_inst_freq, True, True, False) 
	file_score = old_score
	# sys.exit()
	# theta = 0.001
	theta = np.pi/9

	# chosens = [18,26]
	chosens = [11,18,8,38,26,41,10,35,5,34,43,22,0,14]

	for i in trange(int(2*np.pi/theta)+1):
		if iteration%50000 == 0:
			print("saving")
			save_csv(coords)

		total_iterations += 1
		iteration += 1
		mid = np.zeros(2)
		for chosen in chosens:
			mid += coords[chosen]
		mid = mid/len(chosens)

		for chosen in chosens:
			pt = np.array(rotate(mid,coords[chosen],theta))
			coords[chosen] = pt
		print(score(coords,wind_inst_freq, False, False, False) - old_score)
		import pdb
		pdb.set_trace()
		if(score(coords,wind_inst_freq, False, False, False) - old_score > 0):
			import pdb
			pdb.set_trace()
			save_csv_path(coords,'data/rotate_prac.csv')
		

		# if best_ind == -1:
		# 	# print("Chose windmill {} but no improvement in this direction; happened {} consecutive times before this".format(chosen, iters_with_no_inc))
		# 	iters_with_no_inc += 1



		# else:
		# 	print("Chose windmill {} and got an improvement of {} units in the average AEP".format(chosen, best_score - old_score))
		# 	print("Total iter num: {} ".format(total_iterations))
		# 	iters_with_no_inc = 0 #because we are considering such consecutive iters	
		# 	# score(chosen, )
		# 	coords[chosen][0], coords[chosen][1] = possibilities[best_ind]
		# 	old_score = best_score
		# 	original_deficit = best_deficit
		# 	print("average : {}".format(old_score))
		# 	print()

	print("DONE")		 
	save_csv(coords)
