# utils.py
from evaluate import checkConstraints, binWindResourceData, getAEP, loadPowerCurve, getTurbLoc, preProcessing, delta_AEP
import numpy as np
from constants import *
import random

def initialise_valid():
	#for now return the same everytime to compare methods etc
	# return getTurbLoc('../data/turbine_loc_test.csv')
	# this can be done quicker with matrix operations
	# 50 <= x,y <= 3950
	# define an 9x9 grid, choose 50 out of 81 boxes to fill
	# choosing pts in 100, 3800 to give them room on the boundaries as well

	# jump = 3800/9 # >400
	# region = 3800/9 - 400
	region = (3900 - 400*8)/9
	# region = (3800 - 400*8)/9
	jump = 400 + region

	data = []
	# generate 81 points then choose just 50
	for i in range(9):
		# s_x = 100 + (jump)*i
		s_x = 50 + (jump)*i
		for j in range(9):
			s_y = 50 + jump*j
			
			x = s_x + np.random.uniform(0, region)
			y = s_y + np.random.uniform(0, region)

			data.append((x,y))

	random.shuffle(data)
	return np.array(data[:50])

def initialise_periphery():
	data = []
	delta = (3900 - 2*300)/8

	#36 pts on the periphery
	for i in range(9):
		data.append((50 + 300 + delta*i,50 + 1) )
		data.append((50 + 300 + delta*i,3950 - 1) )
		data.append((50 +1, 50 + 300 + delta*i) )
		data.append((3950 - 1,50 + 300 + delta*i) )

	#now fit 16 points in the centre
	# margin = 402 #atleast itna, and less than 1200 ish
	margin = 1000
	inner_dis = (3900 - margin*2)/3
	for i in range(4):
		for j in range(4):
			x = 50 + margin + i*inner_dis
			y = 50 + margin + j*inner_dis
			data.append((x,y)) 

	# random.shuffle(data)
	return np.array(data[:50])


def score(coords, wind_inst_freq, to_print = False, with_deficit = False, get_each_turbine_power=False):
	# success = checkConstraints(coords, turb_diam)
	# if not success:
	# 	print("CONSTRAINTS VIOLATED")
	# 	raise "VIOLATION"
	# 	return MINIMUM 
	ret = getAEP(turb_rad, coords, power_curve, wind_inst_freq, 
		n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t, get_each_turbine_power=get_each_turbine_power, with_deficit=with_deficit) 

	if to_print:
		if with_deficit or get_each_turbine_power:
			AEP, turbine_powers, deficit = ret
		else:
			AEP = ret
		print('Average AEP over train years is {}', "%.12f"%(AEP), 'GWh')
	return ret

def delta_score(coords, wind_inst_freq, chosen, new_x, new_y, original_deficit):
	return delta_AEP(turb_rad, coords, power_curve, wind_inst_freq, 
	            n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t,
	            chosen, new_x, new_y, original_deficit)

def min_dist_from_rest(chosen, coords, new_x, new_y):
	min_d = min([(new_x - coords[i][0])**2 + (new_y - coords[i][1])**2 if i!=chosen else MAXIMUM for i in range(50)])
	min_d = min_d**0.5
	return min_d