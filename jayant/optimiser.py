import numpy as np
from scipy.optimize import minimize
import numpy as np
from evaluate import checkConstraints, binWindResourceData, getAEP, loadPowerCurve, getTurbLoc, preProcessing, getAEP_for_optimiser
from datetime import datetime
import random
from args import make_args
from tqdm import tqdm
from utils import score, initialise_valid, initialise_periphery, min_dist_from_rest, delta_score, initialise_max
from constants import *
from utils import *



# def objective(x):
#     return x[0]*x[3]*(x[0]+x[1]+x[2])+x[2]

# def constraint1(x):
#     return x[0]*x[1]*x[2]*x[3]-25.0

# def constraint2(x):
#     sum_eq = 40.0
#     for i in range(4):
#         sum_eq = sum_eq - x[i]**2
#     return sum_eq

# # initial guesses
# n = 4
# x0 = np.zeros(n)
# x0[0] = 1.0
# x0[1] = 5.0
# x0[2] = 5.0
# x0[3] = 1.0

# # show initial objective
# print('Initial Objective: ' + str(objective(x0)))

# # optimize
# b = (1.0,5.0)
# bnds = (b, b, b, b)
# con1 = {'type': 'ineq', 'fun': constraint1} 
# con2 = {'type': 'eq', 'fun': constraint2}
# cons = ([con1,con2])
# solution = minimize(objective,x0,method='SLSQP',\
#                     bounds=bnds,constraints=cons)
# x = solution.x

# # show final objective
# print('Final Objective: ' + str(objective(x)))

# # print solution
# print('Solution')
# print('x1 = ' + str(x[0]))
# print('x2 = ' + str(x[1]))
# print('x3 = ' + str(x[2]))
# print('x4 = ' + str(x[3]))


#my version
years = [2007, 2008, 2009, 2013, 2014, 2015, 2017]
# years = [2007]
year_wise_dist = np.array([binWindResourceData('../data/WindData/wind_data_{}.csv'.format(year)) for year in years])
wind_inst_freq = np.mean(year_wise_dist, axis = 0)

def my_objective(x):
	return getAEP_for_optimiser(turb_rad, x.reshape(50,2), power_curve, wind_inst_freq, 
		n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t, False)

def my_objective_2(x):
    coords = x.reshape(50,2)
    cost = 0
    for i in range(50):
        for j in range(50):
            cost += np.linalg.norm(coords[i] - coords[j])

    return -cost/2500
    # return sum([(new_x - coords[i][0])**2 + (new_y - coords[i][1])**2 if i!=chosen else MAXIMUM for i in range(50)])
b = [(50, 3950)]*100


objective = my_objective

coords = initialise_valid()
x0 = coords.reshape(-1)
cons = []

# for i in range(50):
# 	for j in range(i+1, 50):
# 		temp = {'type':'ineq', 'fun': lambda x : (x[2*i] - x[2*j])**2 + (x[2*i +1] - x[2*j+1])**2 - 400**2}
# 		cons.append(temp)
def constraint(x):
    a = min_dis(x.reshape(50,2))
    if a >= 400:
        return 1
    else:
        return -1


print('Initial Objective: ' + str(objective(x0)))

solution = minimize(objective,x0,method='SLSQP',\
                    bounds=b,constraints={ 'type' : 'ineq', 'fun':constraint
                    })
x = solution.x
print(solution.fun)
# print()
score(x.reshape(50,2), wind_inst_freq, True)