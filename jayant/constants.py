# constants.py

from evaluate import loadPowerCurve, preProcessing
import numpy as np

MAXIMUM = 10**10
MINIMUM = -10**10
PRECISION = 10e-10

turb_specs    =  {   
                     'Name': 'Anon Name',
                     'Vendor': 'Anon Vendor',
                     'Type': 'Anon Type',
                     'Dia (m)': 100,
                     'Rotor Area (m2)': 7853,
                     'Hub Height (m)': 100,
                     'Cut-in Wind Speed (m/s)': 3.5,
                     'Cut-out Wind Speed (m/s)': 25,
                     'Rated Wind Speed (m/s)': 15,
                     'Rated Power (MW)': 3
                 }
turb_diam      =  turb_specs['Dia (m)']
turb_rad       =  turb_diam/2 
power_curve   =  loadPowerCurve('../data/power_curve.csv')

n_turbs       =   50
    
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

n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t = preProcessing(power_curve)

cos_dir1 = cos_dir[[i*n_slices_sped for i in range(n_slices_drct)]]
sin_dir1 = sin_dir[[i*n_slices_sped for i in range(n_slices_drct)]]
C_t1 = C_t[[i*n_slices_sped for i in range(n_slices_drct)], :, :]
C_t_direct = 1-np.sqrt(1-C_t[:,:,0])