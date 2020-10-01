# test_fetch_movable_segments.py
# import numpy as np
# from scipy.optimize import minimize
import numpy as np
# from evaluate import checkConstraints, binWindResourceData, getAEP, loadPowerCurve, getTurbLoc, preProcessing, getAEP_for_optimiser
# from datetime import datetime
import random
# from args import make_args
# from tqdm import tqdm
from utils import score, initialise_valid, initialise_periphery, min_dist_from_rest, delta_score, initialise_max
from constants import *
from utils import *
from matplotlib import pyplot as plt
import time
coords = initialise_valid()
# direction = np.pi/4

for _ in range(1000000):
	chosen = np.random.randint(0,50)
	direction = np.random.uniform(0, 2*np.pi)
	print("dir considered is {} degrees".format(direction*(180/np.pi)))
	segments = fetch_movable_segments(coords, chosen, direction)
	samples = np.random.uniform(size = len(segments))
	possibilities = [((0.0001*a[0]+ 0.9999*b[0]), (0.0001*a[1] + 0.9999*b[1])) for a,b in segments]
	x = coords[:,0]
	y = coords[:,1]
	flag = False
	for ind, (new_x, new_y) in enumerate(possibilities):
		if not delta_check_constraints(coords, chosen, new_x, new_y):
			print("ERROR")

			print("min dis " ,min_dist_from_rest(chosen, coords, new_x, new_y))
			print("the seg was {}".format(segments[ind]))
			flag = True
			break

	plt.clf()
	plt.scatter(x, y, color = "blue")
	c = [((a[0]+ b[0])/2, (a[1] + b[1])/2) for a,b in segments]
	possibilities = []
	a = []
	b = []
	for pt1, pt2 in segments:
		possibilities.append(pt1)
		possibilities.append(pt2)
		a.append(pt1)
		b.append(pt2)

	x_p = [px for px,_ in possibilities]
	y_p = [py for _,py in possibilities]
	plt.plot(x_p, y_p, color = "orange")
	plt.scatter([x for x,y in a], [y for x,y in a], color = 'g')
	plt.scatter([x for x,y in b], [y for x,y in b], color = 'r')
	plt.scatter([coords[chosen][0]], [coords[chosen][1]], color = 'purple')
	plt.scatter([x for x,y in c], [y for x,y in c], color = 'y')

	if new_x is not None:
		plt.scatter([new_x],[new_y],color = "black")
	plt.savefig("junk/temp.png")

	if flag:
		sys.exit()
	# time.sleep(2)
# circle1 = plt.Circle((coords[chosen][0], coords[chosen][1]), color = "r")
# circle1=plt.Circle((0,0),.2,color='r')
# plt.gcf().gca().add_artist(circle1)
