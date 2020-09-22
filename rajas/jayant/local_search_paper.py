# paper local search
import numpy as np
import math
from evaluate import checkConstraints, binWindResourceData, getAEP, loadPowerCurve, getTurbLoc, preProcessing
import random
from datetime import datetime
from utils import score, initialise_valid, initialise_periphery, min_dist_from_rest, delta_score, initialise_file
from constants import *
import sys

NN = 2
P = 0.2
NUMTIMESMAX = 10
REDUCTION = 0.9
INC_SIZE = 10
INIT_STD_DIS = (649 - 400)/3
SAVE_FREQ = 100000

def to_radian(x):
    return (x * np.pi)/180.0

def save_csv(coords):
    f = open("submissions/temp_{}_avg_aep_{}_iterations_{}.csv"
        .format(score(coords, wind_inst_freq),iteration, str(datetime.now()).replace(':','')), "w")
    np.savetxt(f, coords, delimiter=',', header='x,y', comments='', fmt='%1.8f')
    f.close()

def checkPossible(vi, v):
    new_x = vi[0] + v[0]
    new_y = vi[1] + v[1]
    if not (50 < new_x < 3950 and 50 < new_y < 3950):
        return False

    min_d = min_dist_from_rest(chosen, coords, new_x, new_y)
    if min_d <= 400 + PRECISION:
       return False
    return True

if __name__ == "__main__":

    years = [2007, 2008, 2009, 2013, 2014, 2015, 2017]
    year_wise_dist = np.array([binWindResourceData('../../data/WindData/wind_data_{}.csv'.format(year)) for year in years])
    wind_inst_freq = np.mean(year_wise_dist, axis = 0) #over all years

    iteration = 0
    if len(sys.argv) > 1:
        coords = initialise_file(sys.argv[1])
    else:
        coords = initialise_periphery()

    old_score, original_deficit = score(coords,wind_inst_freq, True, True)
    std_dis = INIT_STD_DIS
    while(True):
        if iteration%SAVE_FREQ and old_score > 527 == 0:
            print("saving")
            save_csv(coords)
        print()
        print("Total iter num: {}".format(iteration))

        iteration += 1
        chosen = np.random.randint(0,50)

        print("current average : {}".format(old_score))

        x, y = coords[chosen]
        vi = np.array([x,y])
        dists = []
        for i, (x_dash, y_dash) in enumerate(coords):
            if i != chosen:
               dists.append(((x - x_dash)**2 + (y - y_dash)**2, x_dash, y_dash))
        dists.sort()

        v = np.zeros(2)
        for i in range(NN):
            v += np.array([x - dists[i][1], y - dists[i][2]])
        v = v / np.linalg.norm(v)

        theta_v = np.arccos(v[0])
        theta = np.random.normal(theta_v, np.pi/6)
        d = np.random.normal(0, std_dis)

        v = d * np.array([math.cos(theta), math.sin(theta)])

        if np.random.uniform(0,1) < P:
            v = -v

        numtimes = 0
        while (checkPossible(vi, v) == False):
            if numtimes == NUMTIMESMAX:
                break
            v *= REDUCTION
            numtimes +=1
        else:
            print ("Was not able to get a good score for {}".format(chosen))
            continue
        vf = vi + v
        new_x, new_y = vf[0], vf[1]

        new_score, new_deficit = delta_score(coords, wind_inst_freq, chosen, new_x, new_y, original_deficit)

        if new_score >= old_score:
            coords[chosen][0], coords[chosen][1] = new_x, new_y
            improvement = new_score - old_score
            old_score = new_score
            original_deficit = new_deficit
            print("Chose windmill {} and got an improvement of {} units in the average AEP".format(chosen, improvement))
            std_dis += INC_SIZE
        else:
            print("Chose windmill {} and did not get an improvement".format(chosen))
            if std_dis - INC_SIZE > INIT_STD_DIS: 
                std_dis -= INC_SIZE