#andha crude local search
import numpy as np
from evaluate import turb_specs, checkConstraints, binWindResourceData, getAEP, loadPowerCurve, getTurbLoc, preProcessing
from datetime import datetime
MINIMUM = -10**10
EPSILON = 100

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
coords   =  getTurbLoc('../data/turbine_loc_test.csv') #supposed to be a numpy array of shape 50 x 2


DIRECTIONS = 12
DELTA = (2*np.pi)/DIRECTIONS

iteration = 0
while(True):
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

	print("For iteration number {}, ".format(iteration))
	if best_coords is None:
		print("Chose windmill {} but no improvement in any direction".format(chosen))
	else:
		print("Chose windmill {} and got an improvement of {} units in the average AEP".format(chosen, best_improvement))	
		# score(chosen, )
		coords = best_coords

	if iteration%200 == 199:
		f = open("submissions/temp_{}.csv".format(str(datetime.now()).replace(':','')), "w")
		np.savetxt(f, coords, delimiter=',', header='x,y', comments='', fmt='%1.8f')
		f.close()

print("DONE")		 
f = open("submissions/temp_{}.csv".format(str(datetime.now()).replace(':','')), "w")
np.savetxt(f, coords, delimiter=',', header='x,y', comments='', fmt='%1.8f')
f.close()