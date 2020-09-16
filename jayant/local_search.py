#andha crude local search
import numpy as np
from evaluate import turb_specs, checkConstraints, binWindResourceData, getAEP, loadPowerCurve, getTurbLoc, preProcessing
from datetime import datetime
import random

MINIMUM = -10**10
EPSILON = 30
RANDOM_RESTART_THRESH = 100
DIRECTIONS = 12

def save_csv(coords):
	f = open("submissions/temp_{}_avg_aep_{}m_jump_{}_directions_{}_iterations_{}.csv"
		.format(score(coords), EPSILON,DIRECTIONS,iteration, str(datetime.now()).replace(':','')), "w")
	np.savetxt(f, coords, delimiter=',', header='x,y', comments='', fmt='%1.8f')
	f.close()

def initialise_valid():
	# this can be done quicker with matrix operations
	# 50 <= x,y <= 3950
	# define an 9x9 grid, choose 50 out of 81 boxes to fill
	# choosing pts in 100, 3800 to give them room on the boundaries as well

	jump = 3800/9 # >400
	region = 3800/9 - 400

	data = []
	# generate 81 points then choose just 50
	for i in range(9):
		s_x = 100 + (jump)*i
		for j in range(9):
			s_y = 100 + jump*j
			
			x = s_x + np.random.uniform(0, region)
			y = s_y + np.random.uniform(0, region)

			data.append((x,y))

	random.shuffle(data)
	return np.array(data[:50])

def score(coords, to_print = False):
	success = checkConstraints(coords, turb_diam)
	if not success:
		return MINIMUM 
	ans = 0
	years = [2007, 2008, 2009, 2013, 2014, 2015, 2017]
	for year in years:
	    wind_inst_freq =  binWindResourceData('../data/WindData/wind_data_{}.csv'.format(year))   
	    AEP = getAEP(turb_rad, coords, power_curve, wind_inst_freq, 
	              n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t) 
	    if to_print:
	    	print('Total power produced by the wind farm in ',year,' would have been: ', "%.12f"%(AEP), 'GWh')
	    ans += AEP
	return ans/len(years)

turb_diam      =  turb_specs['Dia (m)']
turb_rad       =  turb_diam/2 
power_curve   =  loadPowerCurve('../data/power_curve.csv')
n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t = preProcessing(power_curve)
#initialise coords with working values

# coords   =  getTurbLoc('../data/turbine_loc_test.csv') #supposed to be a numpy array of shape 50 x 2
#to check initialiser
# while(True):
# 	coords = initialise_valid()
# 	if not checkConstraints(coords, turb_diam):
# 		print("wrong initialiser")
# 		break

# sys.exit()
coords = initialise_valid()

DELTA = (2*np.pi)/DIRECTIONS

total_iterations = 0
iteration = 0
num_restarts = 0
iters_with_no_inc = 0
while(True):
	if iters_with_no_inc >= RANDOM_RESTART_THRESH:
		save_csv(coords)
		iters_with_no_inc = 0
		iteration = 0
		num_restarts += 1
	total_iterations += 1
	iteration += 1
	chosen = np.random.randint(0,50) #50 is not included
	#now lets see whether we can improve upon our score

	coords = coords.copy()
	x, y = coords[chosen]

	#considering just
	best_coords = None
	best_improvement = 0
	old_score = score(coords, True) 
	print("average : {}".format(old_score))
	print()
	for option in range(DIRECTIONS):
		angle = option*DELTA

		new_x = x + EPSILON*np.cos(angle)
		new_y = y + EPSILON*np.sin(angle)

		copied = coords.copy() # an undo can be done to optimise later
		copied[chosen][0], copied[chosen][1] = new_x, new_y 

		new_score = score(copied)
		improvement = new_score - old_score

		if improvement >= best_improvement:
			best_improvement = improvement
			best_coords = copied
		

	print("Total iter num: {}, num restarts: {}, local iteration number {}, ".format(total_iterations, num_restarts, iteration))
	if best_coords is None:
		print("Chose windmill {} but no improvement in any direction; happened {} consecutive times before this".format(chosen, iters_with_no_inc))
		iters_with_no_inc += 1
	else:
		print("Chose windmill {} and got an improvement of {} units in the average AEP".format(chosen, best_improvement))
		iters_with_no_inc = 0 #because we are considering such consecutive iters	
		# score(chosen, )
		coords = best_coords

	if iteration%200 == 199:
		save_csv(coords)

print("DONE")		 
save_csv(coords)