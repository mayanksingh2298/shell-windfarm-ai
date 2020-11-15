import math
from tqdm import tqdm,trange
import os
import sys
import multiprocessing as mp
import numpy as np
import time
from evaluate import checkConstraints, binWindResourceData, getAEP, loadPowerCurve, getTurbLoc, preProcessing
from datetime import datetime
import random
from args import make_args
from tqdm import tqdm
from utils import score, initialise_valid, initialise_periphery, min_dist_from_rest, delta_score, delta_check_constraints, fetch_movable_segments
from utils import min_dis, initialise_file
from constants import *
from datetime import datetime


args = make_args()

#-------hyperparameters---------
NPROCS = 3
NEGATIVE_CHILDREN = 1
RAJAS_CHILDREN = 1
CHILDREN_IMPROVEMENT_THRESHOLD = 0
MASTER_IMPROVEMENT_THRESHOLD = 0
MASTER_REALLY_IMPROVEMENT_THRESHOLD = 0.01
YEARS = [2007, 2008, 2009, 2013, 2014, 2015, 2017]
# YEARS = [2015, 2017]
# negative hyperparams
NEGATIVE_NON_POS_INC_THRESHOLD = 10000
# rajas hyperparams
NN = 2
P = 0.2
NUMTIMESMAX = 10
REDUCTION = 0.9
INC_SIZE = 10
INIT_STD_DIS = (649 - 400)/3


#-------hyperparameters---------
#no nengative processes are reinitialized at MASTER_IMPROVEMENT_THRESHOLD
#all processes are reinitialized at MASTER_REALLY_IMPROVEMENT_THRESHOLD
#initial processes are negative  
WINNER_CSV = "data/winner.csv"
CHILDREN_DATA_ROOT = "data/children"

def save_csv(coords,path):
	f = open(path,"w")
	np.savetxt(f, coords, delimiter=',', header='x,y', comments='', fmt='%1.8f')
	f.close()

def master(wind_inst_freq,master_to_child_qlis,child_to_master_qlis):
	coords = getTurbLoc(WINNER_CSV)
	master_score = score(coords,wind_inst_freq)
	current_time = datetime.now().strftime("%H:%M:%S")
	print(current_time,"- Initial score:",master_score)
	it = 0
	while True:
		it+=1
		try:
			# calculate the score of master.csv
			coords = getTurbLoc(WINNER_CSV)
			master_score = score(coords,wind_inst_freq)

			# get a lis of all children files and find best improved score over master
			children_files = os.listdir(CHILDREN_DATA_ROOT)
			best_ind = -1
			best_score = -1
			for ind, file in enumerate(children_files):
				# get turbine coordinates of child file
				coords = getTurbLoc(os.path.join(CHILDREN_DATA_ROOT,file))

				# score these coordinates
				curr_score = score(coords,wind_inst_freq)
				if(curr_score > master_score):
					save_csv(coords,"data/temp_"+str(curr_score)+"_"+str(it)+".csv")

				if(curr_score > master_score + MASTER_IMPROVEMENT_THRESHOLD and curr_score > best_score):
					# this is a really good file
					best_ind = ind
					best_score = curr_score

			if best_ind==-1: # no improvement has happened
				time.sleep(5) # rest a while baby girl
				continue
			else:
				best_file = os.path.join(CHILDREN_DATA_ROOT,children_files[best_ind])
				best_coords = getTurbLoc(best_file)
				current_time = datetime.now().strftime("%H:%M:%S")
				print(current_time,"- New score:",best_score,"from file:",best_file)
				# write this file to winner.csv
				save_csv(best_coords,WINNER_CSV)


				#send a signal to each non-negative child process
				for child in range(NEGATIVE_CHILDREN,NPROCS):
					master_to_child_qlis[child].put(1)
				#send a signal to negative children if a really good improvement
				if(best_score>=master_score+MASTER_REALLY_IMPROVEMENT_THRESHOLD):
					print("Really good!")
					for child in range(NEGATIVE_CHILDREN):
						master_to_child_qlis[child].put(1)

				#wait for ack from each non-negative child process before proceeding again
				for child in range(NEGATIVE_CHILDREN,NPROCS):
					child_to_master_qlis[child].get()
				#wait for ack from negative children, if really good improvement
				if(best_score>=master_score+MASTER_REALLY_IMPROVEMENT_THRESHOLD):
					for child in range(NEGATIVE_CHILDREN):
						child_to_master_qlis[child].get()
		except:
			continue
					


def local_search(wind_inst_freq,child,master_to_child_qlis,child_to_master_qlis):
	f = open(os.path.join("data",str(child)+".log"),'w')
	while True:
		coords = getTurbLoc(WINNER_CSV)	
		iteration = -1
		total_iterations = 0
		num_restarts = 0
		iters_with_no_inc = 0
		old_score, original_deficit = score(coords,wind_inst_freq, False, True, False) 
		file_score = old_score
		# sys.exit()
		while(True):
		# for i in trange(10000000000):
			#before beginning iteration check for signal from master
			if(not master_to_child_qlis[child].empty()):
				master_to_child_qlis[child].get()
				# send ack to master
				child_to_master_qlis[child].put(1)
				break

			total_iterations += 1
			iteration += 1
			chosen = np.random.randint(0,50) #50 is not included
			x, y = coords[chosen]
			found = False
			best_ind = -1
			best_score = old_score + CHILDREN_IMPROVEMENT_THRESHOLD

			direction = np.random.uniform(0, 2*np.pi)
			segments = fetch_movable_segments(coords, chosen, direction)
			
			possibilities = []

			samples = np.random.uniform(size = len(segments))
			possibilities += [((samples[i]*a[0]+ (1-samples[i])*b[0]), (samples[i]*a[1] + (1-samples[i])*b[1])) for i,(a,b) in enumerate(segments)]
			
			random.shuffle(possibilities)
			entered = False
			for ind, (new_x, new_y) in enumerate(possibilities):
				if not delta_check_constraints(coords, chosen, new_x, new_y):
					# print("ERROR")
					# sys.exit()
					continue
				entered = True
				new_score, new_deficit = delta_score(coords, wind_inst_freq, chosen, new_x, new_y, original_deficit)

				if new_score >= best_score:
					best_score = new_score
					best_ind = ind
					best_deficit = new_deficit


			if best_ind == -1:
				iters_with_no_inc += 1
			else:
				# print("Chose windmill {} and got an improvement of {} units in the average AEP".format(chosen, best_score - old_score))
				# print("Total iter num: {} ".format(total_iterations))
				iters_with_no_inc = 0 #because we are considering such consecutive iters	
				# score(chosen, )
				coords[chosen][0], coords[chosen][1] = possibilities[best_ind]
				if(best_score>old_score):
					# save csv to disk only if actual improvement
					save_csv(coords,os.path.join(CHILDREN_DATA_ROOT,"{}.csv".format(child)))
				old_score = best_score
				original_deficit = best_deficit
				print(child,iteration,best_score,file=f)
				f.flush()

				# print("average : {}".format(old_score))

				# print()

def local_search_negative(wind_inst_freq,child,master_to_child_qlis,child_to_master_qlis):
	f = open(os.path.join("data",str(child)+".log"),'w')
	while True:
		coords = getTurbLoc(WINNER_CSV)
		iteration = -1
		total_iterations = 0
		num_restarts = 0
		iters_with_no_inc = 0
		old_score, original_deficit = score(coords,wind_inst_freq, False, True, False) 
		global_best_score = old_score 
		iters_with_non_pos_increase = 0
		while(True):
			#before beginning iteration check for signal from master
			if(not master_to_child_qlis[child].empty()):
				master_to_child_qlis[child].get()
				# send ack to master
				child_to_master_qlis[child].put(1)
				break

			total_iterations += 1
			iteration += 1
			chosen = np.random.randint(0,50) #50 is not included
			x, y = coords[chosen]
			found = False
			best_ind = -1
			best_score = old_score + CHILDREN_IMPROVEMENT_THRESHOLD

			direction = np.random.uniform(0, 2*np.pi)
			segments = fetch_movable_segments(coords, chosen, direction)
			
			possibilities = []
			#uniform samples from each seg
			samples = np.random.uniform(size = len(segments))
			possibilities += [((samples[i]*a[0]+ (1-samples[i])*b[0]), (samples[i]*a[1] + (1-samples[i])*b[1])) for i,(a,b) in enumerate(segments)]

			entered = False
			random.shuffle(possibilities)
			for ind, (new_x, new_y) in enumerate(possibilities):
				if not delta_check_constraints(coords, chosen, new_x, new_y):
					continue
				entered = True
				new_score, new_deficit = delta_score(coords, wind_inst_freq, chosen, new_x, new_y, original_deficit)
				if new_score >= best_score:
					best_score = new_score
					best_ind = ind
					best_deficit = new_deficit
			if best_ind == -1:
				iters_with_no_inc += 1
				iters_with_non_pos_increase += 1
				if iters_with_non_pos_increase > NEGATIVE_NON_POS_INC_THRESHOLD and entered:
					# print("going mad")
					# if old_score >= file_score - 0.1:
					# 	save_csv(coords)
					coords[chosen][0], coords[chosen][1] = new_x, new_y
					old_score = new_score
					original_deficit = new_deficit
					iters_with_non_pos_increase = 0

			else:
				iters_with_no_inc = 0 #because we are considering such consecutive iters	
				coords[chosen][0], coords[chosen][1] = possibilities[best_ind]
				if(best_score>global_best_score):
					save_csv(coords,os.path.join(CHILDREN_DATA_ROOT,"{}.csv".format(child)))
					#print("child:{} global_best_score:{}".format(child,best_score))
					global_best_score = best_score

				if (best_score - old_score) == 0:
					iters_with_non_pos_increase += 1
				else:
					iters_with_non_pos_increase = 0
				old_score = best_score
				original_deficit = best_deficit
				print(child,iteration,best_score,file=f)
				f.flush()
def local_search_rajas(wind_inst_freq,child,master_to_child_qlis,child_to_master_qlis):
	def checkPossible(vi, v, chosen):
		new_x = vi[0] + v[0]
		new_y = vi[1] + v[1]
		if not (50 <= new_x <= 3950 and 50 <= new_y <= 3950):
			return False

		min_d = min_dist_from_rest(chosen, coords, new_x, new_y)
		if min_d <= 400 + PRECISION:
		   return False
		return True
	f = open(os.path.join("data",str(child)+".log"),'w')
	while True:
		coords = getTurbLoc(WINNER_CSV)
		iteration = -1
		old_score, original_deficit = score(coords,wind_inst_freq, False, True, False)
		std_dis = INIT_STD_DIS


		while(True):
			#before beginning iteration check for signal from master
			if(not master_to_child_qlis[child].empty()):
				master_to_child_qlis[child].get()
				# send ack to master
				child_to_master_qlis[child].put(1)
				break
			iteration += 1

			best_score = old_score + CHILDREN_IMPROVEMENT_THRESHOLD
			best_coords = None
			best_ind = -1
			best_x = -1
			best_y = -1

			for chosen in range(50):
			# chosen = np.random.randint(0,50)
				x, y = coords[chosen]
				vi = np.array([x,y])
				dists = []
				for i, (x_dash, y_dash) in enumerate(coords):
					if i != chosen:
					   dists.append(((x - x_dash)**2 + (y - y_dash)**2, x_dash, y_dash))
				dists.sort()

				v = np.zeros(2)
				for i in range(NN):
					v += np.array([x - dists[i][1], y - dists[i][2]])
				v = v / np.linalg.norm(v)

				theta_v = np.arccos(v[0])
				theta = np.random.normal(theta_v, np.pi/6)
				d = np.random.normal(0, std_dis)

				v = d * np.array([math.cos(theta), math.sin(theta)])

				if np.random.uniform(0,1) < P:
					v = -v

				numtimes = 0
				flag = False

				while (checkPossible(vi, v, chosen) == False):
					if numtimes == NUMTIMESMAX:
						flag = True
						break
					v *= REDUCTION
					numtimes +=1
				
				if flag:
					# print ("Was not able to get a good score for {}".format(chosen))
					continue
				vf = vi + v
				new_x, new_y = vf[0], vf[1]

				new_score, new_deficit = delta_score(coords, wind_inst_freq, chosen, new_x, new_y, original_deficit)

				if new_score >= best_score:
					# copied = coords.copy()
					# copied[chosen][0], copied[chosen][1] = new_x, new_y
					# best_coords = copied
					# best_deficit = new_deficit
					# best_mill = chosen
					# best_score = new_score

					best_score = new_score
					best_ind = chosen
					best_deficit = new_deficit
					best_x = new_x
					best_y = new_y






			# if best_coords is not None:
			if best_ind != -1:
				coords[best_ind][0] , coords[best_ind][1] = best_x, best_y
				if(best_score > old_score):
					save_csv(coords,os.path.join(CHILDREN_DATA_ROOT,"{}.csv".format(child)))
				old_score = best_score
				original_deficit = best_deficit
				std_dis += INC_SIZE
				print(child,iteration,best_score,file=f)
				f.flush()

			else:
				# print("Chose windmill {} and did not get an improvement".format(chosen))
				if std_dis - INC_SIZE > INIT_STD_DIS: 
					std_dis -= INC_SIZE


if __name__ == '__main__':
	if args.year is None:
		years = YEARS
	else:
		years = [args.year]
	year_wise_dist = np.array([binWindResourceData('../../data/WindData/wind_data_{}.csv'.format(year)) for year in years])
	wind_inst_freq = np.mean(year_wise_dist, axis = 0) #over all years



	#initialize queues
	master_to_child_qlis = []
	child_to_master_qlis = []
	for i in range(NPROCS):
		master_to_child_qlis.append(mp.Queue())
		child_to_master_qlis.append(mp.Queue())

	for child in range(NPROCS):
		pid = os.fork()
		if(pid==0):
			# child
			if(child<NEGATIVE_CHILDREN):
				local_search_negative(wind_inst_freq,child,master_to_child_qlis,child_to_master_qlis)
			elif(child<RAJAS_CHILDREN+NEGATIVE_CHILDREN):
				local_search_rajas(wind_inst_freq,child,master_to_child_qlis,child_to_master_qlis)
			else:
				local_search(wind_inst_freq,child,master_to_child_qlis,child_to_master_qlis)
	
	#master
	if(pid>0):
		master(wind_inst_freq,master_to_child_qlis,child_to_master_qlis)
