import pandas as pd
import json
import numpy as np
from windrose import WindroseAxes
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import seaborn as sns; sns.set_theme()

import sys
sys.path.append("../evaluator")
from Farm_Evaluator_Vec import binWindResourceData, searchSorted, preProcessing


df = pd.read_csv(sys.argv[1], sep=',', dtype = np.float32)
turb_coords = df.to_numpy(dtype = np.float32)
turb_coords.shape

powerCurve = pd.read_csv('../data/power_curve.csv', sep=',', dtype = np.float32)
powerCurve = powerCurve.to_numpy(dtype = np.float32)
powerCurve.shape


def preProcessing(power_curve, t_zones, n_turbs):
    # number of turbines
#     n_turbs       =   2
    
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
    C_t     = power_curve[indices,1]
    # stacking and reshaping to assist vectorization
    C_t     = np.column_stack([C_t]*(n_turbs*t_zones))
    C_t     = C_t.reshape(n_wind_instances, t_zones, n_turbs)
    
    return(n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t)


farm_size = 3900
delta = 100
t_zones = int((farm_size/delta + 1)*(farm_size/delta + 1))
n_turbs = 2

# df = pd.read_csv('../data/WindData/wind_data_2007.csv')
wind_inst_freq = binWindResourceData('../data/WindData/wind_data_2007.csv')
turb_coords
farm_coords = np.mgrid[50:3955:delta, 50:3955:delta].reshape(2,-1).T
n_turbs = turb_coords.shape[0]
n_farms = farm_coords.shape[0]
n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t = preProcessing(powerCurve, t_zones, n_turbs)
turb_rad = 50

rotate_coords_t   =  np.zeros((n_wind_instances, n_turbs, 2), dtype=np.float32)
rotate_coords_t[:,:,0] =  np.matmul(cos_dir, np.transpose(turb_coords[:,0].reshape(n_turbs,1))) - \
                       np.matmul(sin_dir, np.transpose(turb_coords[:,1].reshape(n_turbs,1)))
rotate_coords_t[:,:,1] =  np.matmul(sin_dir, np.transpose(turb_coords[:,0].reshape(n_turbs,1))) +\
                       np.matmul(cos_dir, np.transpose(turb_coords[:,1].reshape(n_turbs,1)))

rotate_coords_f   =  np.zeros((n_wind_instances, n_farms, 2), dtype=np.float32)
rotate_coords_f[:,:,0] =  np.matmul(cos_dir, np.transpose(farm_coords[:,0].reshape(n_farms,1))) - \
                       np.matmul(sin_dir, np.transpose(farm_coords[:,1].reshape(n_farms,1)))
rotate_coords_f[:,:,1] =  np.matmul(sin_dir, np.transpose(farm_coords[:,0].reshape(n_farms,1))) +\
                       np.matmul(cos_dir, np.transpose(farm_coords[:,1].reshape(n_farms,1)))

tmprx = rotate_coords_t[:,:,0,np.newaxis]
tmpry = rotate_coords_t[:,:,1,np.newaxis]
tmpfx = rotate_coords_f[:,:,0,np.newaxis]
tmpfy = rotate_coords_f[:,:,1,np.newaxis]
x_dist = tmpfx - tmprx.transpose([0,2,1])
y_dist = np.abs(tmpfy - tmpry.transpose([0,2,1]))
sped_deficit = (1-np.sqrt(1-C_t))*((turb_rad/(turb_rad + 0.05*x_dist))**2) 
sped_deficit[((x_dist <= 0) | ((x_dist > 0) & (y_dist > (turb_rad + 0.05*x_dist))))] = 0.0    
sped_deficit_eff  = np.sqrt(np.sum(np.square(sped_deficit), axis = 2))
wind_sped_eff = wind_sped_stacked*(1.0-sped_deficit_eff)

indices = searchSorted(powerCurve[:,0], wind_sped_eff.ravel())
power   = powerCurve[indices,2]
power   = power.reshape(n_wind_instances,t_zones)

wind_sped_eff.shape




a = np.matmul(power.transpose(),wind_inst_freq).reshape(int(farm_size/delta + 1),int(farm_size/delta + 1))
print(a.shape)
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')

sns.heatmap(a, linewidth=0, square=True, ax=ax)
ax.scatter(turb_coords[:,0]/delta,turb_coords[:,1]/delta, marker='*', s=100, color='yellow') 
ax.invert_yaxis()
plt.show()


