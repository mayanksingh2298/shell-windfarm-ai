"""
	usage:
		for empty canvas: python3 manual_placing.py 
		load a csv initially: python3 manual_placing.py <path to csv>
"""	
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from evaluate import binWindResourceData, searchSorted
import seaborn as sns
# sns.set_theme()

from utils import *
from constants import *
scale = 10
D = int(400/scale)
class LineBuilder:
	def __init__(self, line,ax,color):
		self.line = line
		self.ax = ax
		self.color = color
		self.xs = []
		self.ys = []
		self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
		self.counter = 0
		self.shape_counter = 0
		self.shape = {}
		self.precision = 10
		self.save_time = str(datetime.now()).replace(':','').replace(" ","")
		self.pointer = {} # this stores the pointer to the art  - it would help us in removing
		self.all_pts = []
		self.cbar = None
		self.scale = scale
		self.farm_size = 3900
		self.delta = 100
		self.t_zones = int((self.farm_size/self.delta + 1)*(self.farm_size/self.delta + 1))
		self.n_turbs = None
		self.farm_coords = np.mgrid[50:3955:self.delta, 50:3960:self.delta].reshape(2,-1).T
		self.n_farms = self.farm_coords.shape[0]
		self.powerCurve = pd.read_csv('../../data/power_curve.csv', sep=',', dtype = np.float32)
		self.powerCurve = self.powerCurve.to_numpy(dtype = np.float32)
		self.turb_rad = 50
		self.n_wind_instances, self.cos_dir, self.sin_dir, self.wind_sped_stacked, self.C_t = self.modified_preProcessing(self.powerCurve, self.t_zones)
		self.rotate_coords_f = np.zeros((self.n_wind_instances, self.n_farms, 2), dtype=np.float32)
		self.rotate_coords_f[:,:,0] = np.matmul(self.cos_dir, np.transpose(self.farm_coords[:,0].reshape(self.n_farms,1))) - \
							   np.matmul(self.sin_dir, np.transpose(self.farm_coords[:,1].reshape(self.n_farms,1)))
		self.rotate_coords_f[:,:,1] = np.matmul(self.sin_dir, np.transpose(self.farm_coords[:,0].reshape(self.n_farms,1))) +\
							   np.matmul(self.cos_dir, np.transpose(self.farm_coords[:,1].reshape(self.n_farms,1)))

		if(len(sys.argv)==2):
			self.load(sys.argv[1])

	def modified_preProcessing(self, power_curve, t_zones):		
		# direction 'slices' in degrees
		slices_drct   = np.roll(np.arange(10, 361, 10, dtype=np.float32), 1)
		## slices_drct   = [360, 10.0, 20.0.......340, 350]
		n_slices_drct = slices_drct.shape[0]
		
		# speed 'slices'
		slices_sped   = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 
							18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0]
		n_slices_sped = len(slices_sped)-1
		
		# number of wind instances
		n_wind_instances = (n_slices_drct)*(n_slices_sped)
		
		# Create wind instances. There are two columns in the wind instance array
		# First Column - Wind Speed. Second Column - Wind Direction
		# Shape of wind_instances (n_wind_instances,2). 
		# Values [1.,360.],[3.,360.],[5.,360.]...[25.,350.],[27.,350.],29.,350.]
		wind_instances = np.zeros((n_wind_instances,2), dtype=np.float32)
		counter = 0
		for i in range(n_slices_drct):
			for j in range(n_slices_sped): 
				
				wind_drct =  slices_drct[i]
				wind_sped = (slices_sped[j] + slices_sped[j+1])/2
				
				wind_instances[counter,0] = wind_sped
				wind_instances[counter,1] = wind_drct
				counter += 1

		# So that the wind flow direction aligns with the +ve x-axis.			
		# Convert inflow wind direction from degrees to radians
		wind_drcts =  np.radians(wind_instances[:,1] - 90)
		# For coordinate transformation 
		cos_dir = np.cos(wind_drcts).reshape(n_wind_instances,1)
		sin_dir = np.sin(wind_drcts).reshape(n_wind_instances,1)
		
		# create copies of n_wind_instances wind speeds from wind_instances
		wind_sped_stacked = np.column_stack([wind_instances[:,0]]*t_zones)
	   
		# Pre-prepare matrix with stored thrust coeffecient C_t values for 
		# n_wind_instances shape (n_wind_instances, n_turbs, n_turbs). 
		# Value changing only along axis=0. C_t, thrust coeff. values for all 
		# speed instances.
		# we use power_curve data as look up to estimate the thrust coeff.
		# of the turbine for the corresponding closest matching wind speed
		indices = searchSorted(power_curve[:,0], wind_instances[:,0])
		C_t = power_curve[indices,1]
		return(n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t)

	def stack_C_t(self):
		# stacking and reshaping to assist vectorization
		C_t = np.column_stack([self.C_t]*(self.n_turbs*self.t_zones))
		C_t = C_t.reshape(self.n_wind_instances, self.t_zones, self.n_turbs)
		
		return C_t

	def heatmap(self, data, ax=None, cbarlabel="", **kwargs):
		if not ax:
			ax = plt.gca()

		# Plot the heatmap
		im = ax.imshow(data, **kwargs)

		# Create colorbar
		if self.cbar is None:
			self.cbar = ax.figure.colorbar(im, ax=ax)
			self.cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
		else:
			self.cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

	def plot_heatmap(self, turb_coords):
		if self.n_turbs is not None:
			C_t = self.stack_C_t()
			turb_rad = self.turb_rad

			rotate_coords_t = np.zeros((self.n_wind_instances, self.n_turbs, 2), dtype=np.float32)
			rotate_coords_t[:,:,0] = np.matmul(self.cos_dir, np.transpose(turb_coords[:,0].reshape(self.n_turbs,1))) - \
								   np.matmul(self.sin_dir, np.transpose(turb_coords[:,1].reshape(self.n_turbs,1)))
			rotate_coords_t[:,:,1] = np.matmul(self.sin_dir, np.transpose(turb_coords[:,0].reshape(self.n_turbs,1))) +\
								   np.matmul(self.cos_dir, np.transpose(turb_coords[:,1].reshape(self.n_turbs,1)))

			tmprx = rotate_coords_t[:,:,0,np.newaxis]
			tmpry = rotate_coords_t[:,:,1,np.newaxis]
			tmpfx = self.rotate_coords_f[:,:,0,np.newaxis]
			tmpfy = self.rotate_coords_f[:,:,1,np.newaxis]
			
			x_dist = tmpfx - tmprx.transpose([0,2,1])
			y_dist = np.abs(tmpfy - tmpry.transpose([0,2,1]))
			sped_deficit = (1-np.sqrt(1-C_t))*((turb_rad/(turb_rad + 0.05*x_dist))**2) 
			sped_deficit[((x_dist <= 0.1) | ((x_dist > 0) & (y_dist > (turb_rad + 0.05*x_dist))))] = 0.0
			sped_deficit_eff  = np.sqrt(np.sum(np.square(sped_deficit), axis = 2))
			wind_sped_eff = self.wind_sped_stacked*(1.0-sped_deficit_eff)

			indices = searchSorted(self.powerCurve[:,0], wind_sped_eff.ravel())
			power   = self.powerCurve[indices,2]
			power   = power.reshape(self.n_wind_instances,self.t_zones)

			a = np.matmul(power.transpose(),wind_inst_freq).reshape(int(self.farm_size/self.delta + 1),int(self.farm_size/self.delta + 1))
			# a = np.flip(a, 0)
			a = a.T
			# maxi = np.max(a)
			a = np.kron(a, np.ones((int(self.delta/self.scale),int(self.delta/self.scale))))
			# print(a)
			# a = a[:,:,np.newaxis].repeat(3,axis=2)
			# a = (a / maxi) * 255
			# a = a.astype(int)
			# a[:,:,1] = 255
			# a[:,:,2] = 255
			self.heatmap(a, ax=self.ax, cmap="YlGn")
			# print(a)
			# a = a[:,:,np.newaxis].repeat(1,1,int(self.farm_size/self.delta + 1))
			# fig = plt.figure()
			# self.ax = fig.add_subplot(111, aspect='equal')
			# self.ax.set_data(a)
			# draw()
			# sns.heatmap(a, linewidth=0, square=True, ax=self.ax)
			# ax.scatter(turb_coords[:,0]/delta,turb_coords[:,1]/delta, marker='*', s=100, color='yellow') 
			# self.ax.invert_yaxis()
			# plt.show()

	def plot(self,x,y, heatmap = True):
		self.counter = self.counter + 1
		self.ax.title.set_text('{} points marked'.format(self.counter))
		self.xs.append(x)
		self.ys.append(y)
		dot_pointer = self.ax.scatter( [x], [y], s=20, color='g')
		# for x,y in zip(self.xs,self.ys):
		circle1 = plt.Circle((x, y), D/2, color='slateblue', alpha = 0.3, fill = False)
		circle_pointer = self.ax.add_artist(circle1)
		# self.ax.plot(self.xs,self.ys,color=self.color)
		print("just marked {}, {}".format(self.scale*x, self.scale*y))
		self.all_pts.append((x,y))
		if (heatmap):
			if 0<self.counter <= 50:
				coords = np.array(self.all_pts)
				self.n_turbs = coords.shape[0]
				coords = self.scale*coords
				self.plot_heatmap(coords)
				self.line.figure.canvas.draw()
				f = open("submissions/manual_partial_{}.csv".format(self.save_time), "w")
				np.savetxt(f, coords, delimiter=',', header='x,y', comments='', fmt='%1.8f')
				f.close()
				full_path = os.path.abspath("submissions/manual_partial_{}.csv".format(self.save_time))
				os.system("cd ../../evaluator/; python3 Farm_Evaluator_Vec.py " + full_path)

			if self.counter == 50:
				s = score(coords, wind_inst_freq, True)
				print(s)
				print("done!")
				f = open("submissions/manual_{}_score_{}.csv".format(s, self.save_time), "w")
				np.savetxt(f, coords, delimiter=',', header='x,y', comments='', fmt='%1.8f')
				f.close()

		return dot_pointer,circle_pointer

	def load(self,path):
		coords = getTurbLoc(path).tolist()
		for i,(x,y) in enumerate(coords):
			x /= self.scale
			y /= self.scale
			flag = i==len(coords)-1
			dot_pointer,circle_pointer = self.plot(x,y,heatmap=flag)
			self.pointer[(x,y)] = (dot_pointer,circle_pointer)
		# if(len(self.all_pts)==50):	 
		# 	print(score(np.array(self.all_pts),wind_inst_freq))



	def __call__(self, event):
		if event.inaxes!=self.line.axes: return

		x = event.xdata
		y = event.ydata
		if min(x,y,int(4000/self.scale)-x, int(4000/self.scale)-y) < int(50/self.scale):
			print("too close to the boundary")
			return
		if self.counter != 0:
			min_d = float("inf")
			for (xi,yi) in zip(self.xs, self.ys):
				min_d = min(min_d, ((x - xi)**2 + (y - yi)**2)**0.5)
			# print("min d is ", min_d)
				if min_d < D:
					print("Deleting point:",xi,yi)
					self.xs.remove(xi)
					self.ys.remove(yi)
					pointers = self.pointer[(xi,yi)]
					pointers[0].remove()
					pointers[1].remove()
					del self.pointer[(xi,yi)]
					self.counter -= 1
					self.all_pts.remove((xi,yi))

					coords = np.array(self.all_pts)
					self.n_turbs = coords.shape[0]
					coords = self.scale*coords
					self.plot_heatmap(coords)
					
					self.line.figure.canvas.draw()
					return
		# if self.counter == 0:
		dot_pointer,circle_pointer = self.plot(x,y)
		self.pointer[(x,y)] = (dot_pointer,circle_pointer) 
		# if(len(self.all_pts)):	 
		# 	print(score(np.array(self.all_pts),wind_inst_freq))

		# print(self.xs, self.ys)

def create_shape_on_image(data,cmap='jet'):
	# def change_shapes(shapes):
	#     new_shapes = {}
	#     for i in range(len(shapes)):
	#         l = len(shapes[i][1])
	#         new_shapes[i] = np.zeros((l,2),dtype='int')
	#         for j in range(l):
	#             new_shapes[i][j,0] = shapes[i][0][j]
	#             new_shapes[i][j,1] = shapes[i][1][j]
	#     return new_shapes
	fig = plt.figure()
	ax = fig.add_subplot(111)
	# ax.set_title('click to include shape markers (10 pixel precision to close the shape)')
	line = ax.imshow(data) 
	ax.set_xlim(0,data.shape[1])
	ax.set_ylim(0,data.shape[0])
	linebuilder = LineBuilder(line,ax,'red')
	# plt.gca().invert_yaxis()
	
	plt.show()
	# new_shapes = change_shapes(linebuilder.shape)
	# return new_shapes


years = [2007, 2008, 2009, 2013, 2014, 2015, 2017]
year_wise_dist = np.array([binWindResourceData('../../data/WindData/wind_data_{}.csv'.format(year)) for year in years])
wind_inst_freq = np.mean(year_wise_dist, axis = 0)


#Can write code here to add some pts manually and then the rest via hand
#will have to make some changes in the plotting code usse pehle - ToDO
# scale = 10
#scaled down by a factor of 10 for faster rendering
img = 255*np.ones((int(4000/scale),int(4000/scale)),dtype='uint')
create_shape_on_image(img)
# print(shapes)