
from utils import *
from constants import *


NUM_DIRECTIONS = n_slices_drct #36
NUM_SPEEDS = n_slices_sped # 15



def wind_vector(direction):
	binned_wind = np.zeros((NUM_DIRECTIONS, NUM_SPEEDS), 
                           dtype = np.float32)

	# for i in range(NUM_DIRECTIONS):
		# if i == direction:
	binned_wind[direction] = (1/NUM_SPEEDS) * np.ones(NUM_SPEEDS)
		# else:
			# binned_wind[i] = np.zeros(NUM_SPEEDS)
	return binned_wind.ravel()

# coords = initialise_periphery()
coords = initialise_valid()
for d in range(NUM_DIRECTIONS):
	wind_dist = wind_vector(d)
	print("direction ", d)
	s = score(coords, wind_dist, True)
