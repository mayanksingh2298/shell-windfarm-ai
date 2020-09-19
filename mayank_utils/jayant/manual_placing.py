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

from utils import *
from constants import *

D = 400
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
		if(len(sys.argv)==2):
			self.load(sys.argv[1])


	def plot(self,x,y):
		self.counter = self.counter + 1
		self.ax.title.set_text('{} points marked'.format(self.counter))
		self.xs.append(x)
		self.ys.append(y)
		dot_pointer = self.ax.scatter( [x], [y], s=20, color='g')
		# for x,y in zip(self.xs,self.ys):
		circle1 = plt.Circle((x, y), D/2, color='slateblue', alpha = 0.3, fill = False)
		circle_pointer = self.ax.add_artist(circle1)
		# self.ax.plot(self.xs,self.ys,color=self.color)
		self.line.figure.canvas.draw()
		print("just marked {}, {}".format(10*x, 10*y))
		self.all_pts.append((x,y))
		if 0<self.counter <= 50:
			coords = np.array(self.all_pts)
			coords = coords
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
		for (x,y) in coords:
			dot_pointer,circle_pointer = self.plot(x,y)
			self.pointer[(x,y)] = (dot_pointer,circle_pointer)
		if(len(self.all_pts)==50):	 
			print(score(np.array(self.all_pts),wind_inst_freq))



	def __call__(self, event):
		if event.inaxes!=self.line.axes: return

		x = event.xdata
		y = event.ydata
		if min(x,y,4000-x, 4000-y) < 50:
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
	ax.set_xlim(0,data[:,:,0].shape[1])
	ax.set_ylim(0,data[:,:,0].shape[0])
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

#scaled down by a factor of 10 for faster rendering
img = 256*np.ones((4000,4000,3),dtype='uint')
create_shape_on_image(img)
# print(shapes)