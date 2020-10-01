from tqdm import tqdm,trange
import os
import sys
import multiprocessing as mp
NPROCS = 2


if __name__ == '__main__':
	queue_list = []
	for i in range(NPROCS):
		queue_list.append(mp.Queue())

	for child in range(NPROCS):
		pid = os.fork()
		if(pid>0):
			#master
			# import pdb
			# pdb.set_trace()
			queue_list[child].put(child)
		else:
			while(queue_list[child].empty()):
				pass
			print("I am child {} and I got message: {}".format(child,queue_list[child].get(False)))
			sys.exit()