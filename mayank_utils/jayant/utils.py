# utils.py
from evaluate import checkConstraints, binWindResourceData, getAEP, loadPowerCurve, getTurbLoc, preProcessing, delta_AEP, delta_deficit
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

#from rajas
def initialise_file(filename):
	return pd.read_csv(filename).to_numpy()

def initialise_periphery_mayank_38():
	data = []
	# d = 283
	# delta = (3900 - 2*d)/8

	#36 pts on the periphery
	for i in range(1,10):
		data.append((50 + i*400,50) )
		data.append((50,50 + i*400) )

		data.append((3950,4000 - 50 - 400*i))
		data.append((4000 - 50 - 400*i,3950))
	data.append((50,50))
	data.append((3950,3950))



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


def initialise_periphery_mayank():
	data = []
	d = 283
	delta = (3900 - 2*d)/8

	#36 pts on the periphery
	for i in range(9):
		data.append((50 + d + delta*i,50) )
		data.append((50 + d + delta*i,3950) )
		data.append((50, 50 + d + delta*i) )
		data.append((3950,50 + d + delta*i) )

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

def initialise_max():
	# import numpy as np
	f = open('optimal_50.txt', 'r')

	M_EPS = 1e-6

	pts = []
	for _ in range(50):
		a = f.readline()
		_, x, y = [float(i) for i in a.split()]
		pts.append((x,y))

	pts = np.array(pts)

	# pts = 2*pts
	pts = pts*(1/(1-(2-M_EPS)*(0.071377103865)))
	# pts = 2*pts
	pts = pts + 0.5
	pts = 3900*pts + 50
	return pts

def score(coords, wind_inst_freq, to_print = False, with_deficit = False):
	success = checkConstraints(coords, turb_diam)
	if not success:
		return MINIMUM 
	ret = getAEP(turb_rad, coords, power_curve, wind_inst_freq, 
		n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t, with_deficit) 
	if with_deficit:
		AEP, deficit = ret
	else:
		AEP = ret

	if to_print:
		print('Average AEP over train years is {}', "%.12f"%(AEP), 'GWh')

	if with_deficit:
		return AEP, deficit
	else:
		return AEP

def delta_score(coords, wind_inst_freq, chosen, new_x, new_y, original_deficit):
	return delta_AEP(turb_rad, coords, power_curve, wind_inst_freq, 
	            n_wind_instances, cos_dir1, sin_dir1, wind_sped_stacked, C_t_direct,
	            chosen, new_x, new_y, original_deficit)

def delta_score2(coords, wind_inst_freq, chosen1, chosen2, new_x1, new_y1, new_x2, new_y2, original_deficit):
	delta_score1, delta_deficit1 = delta_AEP(turb_rad, coords, power_curve, wind_inst_freq, 
	            n_wind_instances, cos_dir1, sin_dir1, wind_sped_stacked, C_t_direct,
	            chosen1, new_x1, new_y1, original_deficit)
	new_coords = coords.copy()
	new_coords[chosen1][0], new_coords[chosen1][1] = new_x1, new_y1
	delta_score2, delta_deficit2 = delta_AEP(turb_rad, new_coords, power_curve, wind_inst_freq, 
	            n_wind_instances, cos_dir1, sin_dir1, wind_sped_stacked, C_t_direct,
	            chosen2, new_x2, new_y2, delta_deficit1)
	return delta_score2, delta_deficit2


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
    	print("perimeter violation")
    	# breakpoint()
    	return False

    pt = np.array([new_x, new_y])
    for i in range(coords.shape[0]):
    	if i != chosen:
    		if np.linalg.norm(coords[i] - pt) <= 400:
    			print("too near")
    			# breakpoint()
    			return False
    return True

def delta_check_constraints2(coords, chosen1, chosen2, new_x1, new_y1, new_x2, new_y2):
    if not (50 < new_x1 < 3950 and 50 < new_y1 < 3950):
    	# print(new_)
    	# print("perimeter violation")
    	# breakpoint()
    	return False
    if not (50 < new_x2 < 3950 and 50 < new_y2 < 3950):
    	# print(new_)
    	# print("perimeter violation")
    	# breakpoint()
    	return False

    pt1 = np.array([new_x1, new_y1])
    pt2 = np.array([new_x2, new_y2])

    if(np.linalg.norm(pt1 - pt2) <= 400):
    	return False

    for i in range(coords.shape[0]):
    	if i != chosen1 and i!=chosen2:
    		if (np.linalg.norm(coords[i] - pt1) <= 400) or (np.linalg.norm(coords[i] - pt2) <= 400):
    			# print("too near")
    			# breakpoint()
    			return False

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
		theta += 1e-10 #its improbable that we get the exact angle zero

	pts = []
	eps = 1e-10
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
			delta = (((400+eps)**2 - dis**2))**0.5 + 2*eps

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