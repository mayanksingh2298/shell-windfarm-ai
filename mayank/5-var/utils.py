# utils.py
from evaluate import checkConstraints, binWindResourceData, getAEP, loadPowerCurve, getTurbLoc, preProcessing, delta_AEP, delta_deficit
import numpy as np
from constants import *
import random
import pandas as pd
from constants import wind_inst_freq

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
	return np.array(data[:50], dtype = 'float32')

#from rajas
def initialise_file(filename):
	# return pd.read_csv(filename).to_numpy()
	df = pd.read_csv(filename, sep=',', dtype = np.float32)
	turb_coords = df.to_numpy(dtype = np.float32)
	return turb_coords

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
	return np.array(data[:50], dtype = 'float32')

def initialise_max():
	# import numpy as np
	f = open('optimal_50.txt', 'r')

	M_EPS = 1e-6

	pts = []
	for _ in range(50):
		a = f.readline()
		_, x, y = [float(i) for i in a.split()]
		pts.append((x,y))

	pts = np.array(pts, dtype = 'float32')

	# pts = 2*pts
	pts = pts*(1/(1-(2-M_EPS)*(0.071377103865)))
	# pts = 2*pts
	pts = pts + 0.5
	pts = 3900*pts + 50
	return pts

def score(coords, wind_inst_freq, to_print = False, with_deficit = False, continuous = False, smooth_shadows = False):
	success = checkConstraints(coords, turb_diam)
	if not success:
		return MINIMUM 
	ret = getAEP(turb_rad, coords, power_curve, wind_inst_freq, 
		n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t, with_deficit, continuous, smooth_shadows) 
	if with_deficit:
		AEP, deficit = ret
	else:
		AEP = ret

	if to_print:
		if continuous:
			print('Approximate ', end = '')
		print('Average AEP over train years is {}', "%.12f"%(AEP), 'GWh')

	if with_deficit:
		return AEP, deficit
	else:
		return AEP


def score_pdfo(coords):
	coords = coords.reshape(-1,2)
	ret = getAEP(turb_rad, coords, power_curve, wind_inst_freq, 
		n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t, False, False, False) 
	return -ret

def score_cma(coords):
	# coords shape is 100
	if(len(coords.shape)==1):
		coords = coords.reshape(-1,2)
		ret = getAEP(turb_rad, coords, power_curve, wind_inst_freq, 
		n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t, False, False, False) 
		return -ret

	# coords shape is n x 100
	lis = []
	for i in range(coords.shape[0]):
		coords1 = coords[i]
		coords1 = coords1.reshape(-1,2)
		ret = getAEP(turb_rad, coords1, power_curve, wind_inst_freq, 
			n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t, False, False, False) 
		lis.append(-ret)
	return lis

def score_pso(coords):
	# coords shape is 100
	# if(len(coords.shape)==1):
	# 	coords = coords.reshape(-1,2)
	# 	ret = getAEP(turb_rad, coords, power_curve, wind_inst_freq, 
	# 	n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t, False, False, False) 
	# 	return -ret

	# coords shape is n x 100
	lis = []
	for i in range(coords.shape[0]):
		coords1 = coords[i]
		coords1 = coords1.reshape(-1,2)
		ret = getAEP(turb_rad, coords1, power_curve, wind_inst_freq, 
			n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t, False, False, False) 
		lis.append(-ret)
	return np.array(lis)

def delta_score(coords, wind_inst_freq, chosen, new_x, new_y, original_deficit,continuous = False,  smooth_shadows = False):
	return delta_AEP(turb_rad, coords, power_curve, wind_inst_freq, 
	            n_wind_instances, cos_dir1, sin_dir1, wind_sped_stacked, C_t_direct,
	            chosen, new_x, new_y, original_deficit, continuous,  smooth_shadows)

def delta_loss(coords, wind_inst_freq, chosen, new_x, new_y, original_deficit):
	return delta_deficit(turb_rad, coords, power_curve, wind_inst_freq, 
	            n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t,
	            chosen, new_x, new_y, original_deficit)

def min_dist_from_rest(chosen, coords, new_x, new_y):
	min_d = min([(new_x - coords[i][0])**2 + (new_y - coords[i][1])**2 if i!=chosen else MAXIMUM for i in range(50)])
	min_d = min_d**0.5
	return min_d

def min_dis(coords):
    return min([min([np.linalg.norm(coords[i] - coords[j]) for i in range(j+1,50)]) for j in range(50-1)])

def ignore_speed(arr):
	return arr[[i*n_slices_sped for i in range(n_slices_drct)]]


def delta_check_constraints(coords, chosen, new_x, new_y):
    if not (50 < new_x < 3950 and 50 < new_y < 3950):
    	# print(new_)
    	# print("perimeter violation")
    	# breakpoint()
    	return False

    temp = coords[chosen].copy()
    coords[chosen] = np.array([1e7, 1e7], dtype = np.float32)

    pt = np.array([new_x, new_y])
    mindy = np.linalg.norm(coords - pt, axis = 1).min()
    coords[chosen] = temp

    if mindy <= 400:
    	return False
    # for i in range(coords.shape[0]):
    # 	if i != chosen:
    # 		if np.linalg.norm(coords[i] - pt) <= 400:
    # 			# print("too near")
    # 			# breakpoint()
    # 			return False
    return True


# def get_from_x(x,y,theta,known):
	# return (known, )

def in_perimeter(x,y):
	return 50<x<3950 and 50<y<3950

def both_coords_arent_wrong(x_l, y_l, x_r, y_r):
	# if max(x_l, x_r) <= 50 or min(x_l, x_r) >= 3950 or max(y_l,y_r) <= 50 or min(y_l, y_r) >= 3950:
		#useless constraint
		# return False

	return in_perimeter(x_l, y_l) or in_perimeter(x_r, y_r)


def fetch_movable_segments(coords, chosen, direction):

	x, y = coords[chosen]
	theta = direction
	if np.cos(theta) == 0 or np.sin(theta) == 1:
		theta += np.float32(1e-10) #its improbable that we get the exact angle zero

	pts = []
	eps = np.float32(1e-6)
	low_lim = 50+eps
	upper_lim = 3950 - eps
	pts.append((low_lim, y + np.tan(theta)*(low_lim - x) ))
	pts.append((upper_lim, y + np.tan(theta)*(upper_lim - x) ))
	pts.append((x + (1/np.tan(theta))*(low_lim - y), low_lim))
	pts.append((x + (1/np.tan(theta))*(upper_lim - y), upper_lim))
	pts.sort()
	left, right = pts[1], pts[2]
	#using the pt and dir, get boundaries
	# can this be a numpy operation
	# where we directly calc projections
	dir_vec = np.array([np.cos(theta), np.sin(theta)])
	constraints = []
	for i in range(coords.shape[0]):
		if i != chosen:
			x1, y1 = coords[i]
			vec = np.array([x - x1, y - y1])
			jump = -np.matmul(dir_vec, vec)

			proj = coords[chosen] + jump*dir_vec
			dis = np.linalg.norm(coords[i] - proj)
			# print(dis, jump)
			if dis >= 400 + eps:
				#safe
				continue
			#problem
			delta = (((400+eps)**2 - dis**2))**np.float32(0.5) + 2*eps

			x_l, y_l = proj - delta*dir_vec
			x_r, y_r = proj + delta*dir_vec
			if x_l > x_r:
				x_l, y_l, x_r, y_r = x_r, y_r, x_l, y_l

			if both_coords_arent_wrong(x_l, y_l, x_r, y_r):
				constraints.append(((x_l, y_l),(x_r, y_r))) #increasing val of x ke basis pe
			# else:
			# 	print("ignored {}, {} to {}, {}".format(x_l, y_l, x_r, y_r))
			# constraints.append()
 
			#check proj distance first
	constraints.sort()
	# print()
	# print(left,right)
	# print(constraints)
	#merge constraints
	if len(constraints) == 0:
		# print()
		return [(left, right)]

	# merged = [constraints[0]]
	#start
	# cleaned_constraints = []

	# merged = [constraints[0]]

	# for i in range(1, len(constraints)):
	# 	(x_prev, _) = constraints[i][1]
	# 	(x_next, _) = constraints[i+1][0]


	ans = []
	if constraints[0][0][0] > left[0]:
		ans.append((left, constraints[0][0]))

	last = constraints[0]

	for i in range(1,len(constraints)):
		# (x_prev, _) = constraints[i][1]
		(x_prev, _) = last[1]
		(x_next, _) = constraints[i][0]

		# if x_prev <= 50:
			# continue

		# if x_


		if x_prev >= x_next:
			# pass #no space in bw
			x_next_next, _ = constraints[i][1]
			if x_next_next > x_prev:
				last = (last[0], constraints[i][1])
		else:
			# print(x_prev, x_next, constraints[i], constraints[i+1])
			ans.append((last[1], constraints[i][0]))
			last = constraints[i]



	rightmost = np.argmax([constraints[i][1][0] for i in range(len(constraints))])
	if constraints[rightmost][1][0] < right[0]:
		ans.append((constraints[rightmost][1], right))

	return ans

def initialise_random():
	pts = np.full((50,2), 1e7, dtype =np.float32)
	counter = 0
	while(counter < 50):
		# while(True):
		sample = np.random.uniform(50, 3950, size =2).astype(np.float32)

		if 50 <= sample[0] <= 3950 and 50 <= sample[1] <= 3950:
			if counter == 0:
				#chill
				# pass
				pts[counter] = sample.reshape(1,-1)
				counter += 1
				print(counter)
			else:
				mindy = np.linalg.norm(pts[:counter] - sample, axis = 1).min()
				if mindy <= 400:
					continue
				else:
					pts[counter] = sample
					counter += 1
					# print(counter)
					# break

	return pts