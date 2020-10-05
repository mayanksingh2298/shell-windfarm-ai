from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from constants import *

D = 40
class LineBuilder:
	def __init__(self, line,ax,color):
		self.line = line
		self.ax = ax
		self.color = color
		self.xs = []
		self.ys = []
		self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
		self.counter = len(all_pts)
		self.shape_counter = 0
		self.shape = {}
		self.precision = 10
		self.save_time = str(datetime.now()).replace(':','')

	def __call__(self, event):
		if event.inaxes!=self.line.axes: return

		x = event.xdata
		y = event.ydata
		if min(x,y,400-x, 400-y) <= 5:
			print("too close to the boundary")
			return
		if self.counter != 0:
			min_d = (min([(x - xi)**2 + (y - yi)**2 for (xi,yi) in zip(self.xs, self.ys)]))**0.5
			print("min d is ", min_d)
			if min_d <= D:
				print("tooooo close, try again")
				return
		# if self.counter == 0:
		self.counter = self.counter + 1
		self.ax.title.set_text('{} points marked'.format(self.counter))
		self.xs.append(x)
		self.ys.append(y)
		# if np.abs(event.xdata-self.xs[0])<=self.precision and np.abs(event.ydata-self.ys[0])<=self.precision and self.counter != 0:
			# self.xs.append(self.xs[0])
			# self.ys.append(self.ys[0])
			# self.ax.scatter(self.xs,self.ys,s=120,color=self.color)
			# self.ax.scatter(self.xs[0],self.ys[0],s=80,color='blue')
			# self.ax.plot(self.xs,self.ys,color=self.color)
			# self.line.figure.canvas.draw()
			# self.shape[self.shape_counter] = [self.xs,self.ys]
			# self.shape_counter = self.shape_counter + 1
			# self.xs = []
			# self.ys = []
			# self.counter = 0
		# else:
		# if self.counter != 0:

		self.ax.scatter( [x], [y], s=20, color='g')
		# for x,y in zip(self.xs,self.ys):
		circle1 = plt.Circle((x, y), D/2, color='slateblue', alpha = 0.3, fill = False)
		self.ax.add_artist(circle1)
		# self.ax.plot(self.xs,self.ys,color=self.color)
		self.line.figure.canvas.draw()
		print("just marked {}, {}".format(10*x, 10*y))
		all_pts.append((x,y))
		if 0<self.counter <= 50:
			coords = np.array(all_pts)
			coords = 10*coords
			f = open("submissions/manual_partial_{}.csv".format(self.save_time), "w")
			np.savetxt(f, coords, delimiter=',', header='x,y', comments='', fmt='%1.8f')
			f.close()

		if self.counter == 50:
			s = score(coords, wind_inst_freq, True)
			print(s)
			print("done!")
			f = open("submissions/manual_{}_score_{}.csv".format(s, self.save_time), "w")
			np.savetxt(f, coords, delimiter=',', header='x,y', comments='', fmt='%1.8f')
			f.close()

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
year_wise_dist = np.array([binWindResourceData('../data/WindData/wind_data_{}.csv'.format(year)) for year in years])
wind_inst_freq = np.mean(year_wise_dist, axis = 0)

all_pts = []

#Can write code here to add some pts manually and then the rest via hand
#will have to make some changes in the plotting code usse pehle - ToDO

#scaled down by a factor of 10 for faster rendering
img = 256*np.ones((400,400,3),dtype='uint')
create_shape_on_image(img)
# print(shapes)