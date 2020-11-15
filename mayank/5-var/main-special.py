import numpy as np
from tqdm import trange
import random
import matplotlib.pyplot as plt
import numpy as np
from evaluate import checkConstraints, binWindResourceData, getAEP, loadPowerCurve, getTurbLoc, preProcessing
from datetime import datetime
import random
from args import make_args
from tqdm import tqdm
from utils import score, initialise_valid, initialise_periphery, min_dist_from_rest, delta_score, delta_check_constraints, fetch_movable_segments
from utils import min_dis, initialise_file
from constants import *

def save_csv(coords):
	f = open("submissions-special/temp_{}_avg_aep_{}_iterations_{}.csv"
		.format(round(score(coords, wind_inst_freq),6),iteration, str(datetime.now()).replace(':','')), "w")
	np.savetxt(f, coords, delimiter=',', header='x,y', comments='', fmt='%1.8f')
	f.close()

Dmin = 400+1e-5
W = 4000
H2 = 4
H3 = 4
MIN_THRESH = 0
RANDOM_JUMP_THRES = 800
def plot(coords):
	N = len(coords)

	fig, ax = plt.subplots(figsize=(8,8))

	n = list(range(N))

	ax.set_aspect('equal', 'box')

	x = coords[:,0]
	y = coords[:,1]
	ax.scatter(x, y, c='blue')

	plt.show()

def calculate_minD(coords):
	mind = 999999999
	for i in range(coords.shape[0]):
		for j in range(coords.shape[0]):
			if (i==j):
				continue
			mind = min(mind, np.linalg.norm(coords[i] - coords[j]))
			# if mind<400:
			# 	print(coords[i],coords[j])
			# 	return mind
	return mind
def calculate_minD_special(coords):
	mind = 999999999
	for i in range(40):
		for j in range(40,50):
			if (i==j):
				continue
			mind = min(mind, np.linalg.norm(coords[i] - coords[j]))
			# if mind<400:
			# 	print(coords[i],coords[j])
			# 	return mind
	return mind

def generate_peri(var):
	x = var[0]
	y = var[1]
	coords = []
	for i in range(6):
		# bottom row 
		coords.append([50+i*x,50])
		if(i!=5):
			coords.append([50+x/2+i*x,50+y])
		# top row
		coords.append([50+i*x,3950])
		if(i!=5):
			coords.append([50+x/2+i*x,3950-y])
	for i in range(5):
		#left row
		if(i!=0):
			coords.append([50,50+i*x])
		coords.append([50+y,50+x/2+i*x])
		#right row
		if(i!=0):
			coords.append([3950,50+i*x])
		coords.append([3950-y,50+x/2+i*x])
	return np.array(coords,dtype = np.float32)


def generate(var,subset=50):
	x1 = var[0]
	x2 = var[1]
	x3 = var[2]
	x4 = var[3]
	x5 = var[4]

	#generate a list of coords using x1 and x2
	Dx = Dmin + ((0.2*x1)**H2)*(W - Dmin)
	Dy = Dmin + ((0.2*x2)**H3)*(W - Dmin)
	init_x = 2000
	init_y = 2000
	coords = [[init_x,init_y]]
	while(init_x+Dx<=8000):
		init_y = 2000
		init_x+=Dx
		coords.append([init_x,init_y])
		while(init_y+Dy<=8000):
			init_y+=Dy
			coords.append([init_x,init_y])
		init_y = 2000
		while(init_y-Dy>=-4000):
			init_y-=Dy
			coords.append([init_x,init_y])
	init_x = 2000
	init_y = 2000
	while(init_x+Dx>=-4000):
		init_y = 2000
		init_x-=Dx
		coords.append([init_x,init_y])
		while(init_y+Dy<=8000):
			init_y+=Dy
			coords.append([init_x,init_y])
		init_y = 2000
		while(init_y-Dy>=-4000):
			init_y-=Dy
			coords.append([init_x,init_y])
	coords = np.array(coords, dtype = np.float32)
	
	#rotate
	theta = -np.pi + x3 * 2 * np.pi
	coords = coords - [2000,2000]
	coords = [2000,2000] + np.matmul(coords,[[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])

	#shift
	shift_x = (0.5 + 0.2*x4)*W
	shift_y = (0.5 + 0.2*x5)*W
	coords = coords + [shift_x,shift_y]

	#remove all infeasible turbines
	final_coords = []
	for i in coords:
		if i[0]>=540 and i[0]<=4000-540 and i[1]>=540 and i[1]<= 4000-540:
			final_coords.append(i)
	final_coords = np.array(final_coords, dtype = np.float32)
	np.random.shuffle(final_coords)
	return final_coords[:subset]



if __name__=="__main__":
	min_thresh_to_move = MIN_THRESH
	years = [2007, 2008, 2009, 2013, 2014, 2015, 2017]
	year_wise_dist = np.array([binWindResourceData('../../data/WindData/wind_data_{}.csv'.format(year)) for year in years])
	wind_inst_freq = np.mean(year_wise_dist, axis = 0) #over all years
	var_peri = [780,88.8829441]

	coords_peri = generate_peri(var_peri)
	while True:
		x1 = random.random()
		x2 = random.random()
		x3 = random.random()
		x4 = random.random()
		x5 = random.random()
		var = [x1,x2,x3,x4,x5]
		coords_interior = generate(var,subset = 50-len(coords_peri))
		coords = np.concatenate((coords_peri,coords_interior))
		if(calculate_minD_special(coords)>=400):
			break
	# while True:
	# 	coords = generate(var)
	# 	plot(coords)

	old_score = score(coords,wind_inst_freq, False, False, False)
	print("Initial score:",old_score)

	iteration = -1
	iters_without_improv = -1

	for ii in trange(999999999999999):
		# if iteration%5000 == 0:
		# 	save_csv(new_coords)

		iteration += 1
		iters_without_improv += 1
		if(iters_without_improv>=RANDOM_JUMP_THRES):
			iters_without_improv = 0
			while True:
				choice = random.randint(0,4)
				var[choice] = random.random()
				coords_interior = generate(var, subset = 50-len(coords_peri))
				if(len(coords_interior)==0):
					continue
				coords = np.concatenate((coords_peri,coords_interior))
				# if(coords.shape[0]==50 and calculate_minD_special(coords)>=400):
				if(coords.shape[0]==50):
					break	
			old_score = score(coords,wind_inst_freq, False, False, False)
			print("NEW SCORE after random jump:",old_score)
		while True:
			choice = random.randint(0,4)
			new_var = var.copy()
			new_var[choice]+=np.random.normal()
			new_coords_interior = generate(new_var, subset = 50-len(coords_peri))
			if(len(new_coords_interior)==0):
				continue
			new_coords = np.concatenate((coords_peri,new_coords_interior))
			# if(new_coords.shape[0]==50 and calculate_minD_special(new_coords)>=400):
			if(new_coords.shape[0]==50):
				break

		new_score = score(new_coords, wind_inst_freq, False, False, False)
		if(new_score>=old_score):
			if(new_score>old_score):
				iters_without_improv=0
			var = new_var
			old_score = new_score
			if(new_score>=517):
				save_csv(new_coords)
				RANDOM_JUMP_THRES=9999999999999999999999
			print("New score: {} Iteration: {}".format(str(new_score),str(iteration)))


	
