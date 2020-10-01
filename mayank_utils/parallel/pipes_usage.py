from tqdm import tqdm,trange
import os
import sys
NPROCS = 2


if __name__ == '__main__':
	pipe_list = []
	for i in range(NPROCS):
		pipe_list.append(os.pipe())

	for child in range(NPROCS):
		pid = os.fork()
		if(pid>0):
			#master
			os.close(pipe_list[child][0])
			import pdb
			pdb.set_trace()
			w = os.fdopen(pipe_list[child][1], 'w')
			w.write("hi child"+str(child))
			w.close()
		else:
			os.close(pipe_list[child][1])
			r = os.fdopen(pipe_list[child][0])
			# print(r)
			print("I am child {} and I got message: {}".format(child,r.read()))
			sys.exit()