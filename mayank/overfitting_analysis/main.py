import os
import sys
sys.path.append(sys.path[0]+"/../parallel/")
from utils import score
from evaluate import getTurbLoc,binWindResourceData
from constants import *
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
YEARS = [2007, 2008, 2009, 2013, 2014, 2015, 2017]
COLORS = ['red', 'blue', 'yellow', 'black', 'green', 'purple', 'orange']

def get_wind(years):
	year_wise_dist = np.array([binWindResourceData('../../data/WindData/wind_data_{}.csv'.format(year)) for year in years])
	wind_inst_freq = np.mean(year_wise_dist, axis = 0) #over all years
	return wind_inst_freq


DATA = "data"
if __name__ == '__main__':
	files = os.listdir(DATA)
	scores = [] 
	"""
		scores has data as [
							[x0,x1,x2,x3,x4,x5,x6,x7],
							...
						]
		x0 - all years
		x1 - 2007
		x2 - 2008
		...
	"""
	fig, axs = plt.subplots(2, 4)
	axs[0, 0].set_title('2007')
	axs[0, 1].set_title('2008')
	axs[0, 2].set_title('2009')
	axs[0, 3].set_title('2013')
	axs[1, 0].set_title('2014')
	axs[1, 1].set_title('2015')
	axs[1, 2].set_title('2017')


	for file in tqdm(files[:399999]):
		coords = getTurbLoc(os.path.join(DATA,file))
		tmp = [score(coords,get_wind(YEARS))]
		for year in YEARS:
			tmp.append(score(coords,get_wind([year])))
		scores.append(tmp)

	scores.sort()
	scores = np.array(scores)
	# for i,year in enumerate(YEARS):
	# 	plt.plot(scores[:,0], scores[:,i+1], color=COLORS[i], marker='o', label=YEARS[i])
	axs[0,0].plot(scores[:,0], scores[:,1], marker='o')
	axs[0,1].plot(scores[:,0], scores[:,2], marker='o')
	axs[0,2].plot(scores[:,0], scores[:,3], marker='o')
	axs[0,3].plot(scores[:,0], scores[:,4], marker='o')
	axs[1,0].plot(scores[:,0], scores[:,5], marker='o')
	axs[1,1].plot(scores[:,0], scores[:,6], marker='o')
	axs[1,2].plot(scores[:,0], scores[:,7], marker='o')

	plt.grid(True)
	# plt.legend()
	plt.show()


