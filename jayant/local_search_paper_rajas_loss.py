# paper local search
import numpy as np
import math
from evaluate import checkConstraints, binWindResourceData, getAEP, loadPowerCurve, getTurbLoc, preProcessing
import random
from datetime import datetime
from utils import score, initialise_valid, initialise_periphery, min_dist_from_rest, delta_loss, initialise_file, ignore_speed
from constants import *
import sys
import time

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

def checkPossible(vi, v, chosen):
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
    year_wise_dist = np.array([binWindResourceData('../data/WindData/wind_data_{}.csv'.format(year)) for year in years])
    wind_inst_freq = np.mean(year_wise_dist, axis = 0) #over all years

    iteration = 0
    if len(sys.argv) > 1:
        coords = initialise_file(sys.argv[1])
    else:
        coords = initialise_periphery()

    _, original_deficit = score(coords,wind_inst_freq, True, True)
    # original_deficit = ignore_speed(original_deficit)
    old_loss = np.sum(np.matmul(wind_inst_freq, original_deficit))
    std_dis = INIT_STD_DIS


    while(True):
        if iteration%SAVE_FREQ == 0:
            print("saving")
            save_csv(coords)

        if iteration%10000 == 0:
            score(coords,wind_inst_freq, True, True)
            time.sleep(0.5)

        iteration += 1
        chosen = np.random.randint(0,50)

        if iteration%10000 == 0:
            print()
            print("Total iter num: {}".format(iteration))


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
        flag = False
        while (checkPossible(vi, v, chosen) == False):
            if numtimes == NUMTIMESMAX:
                flag = True
                break
            v *= REDUCTION
            numtimes +=1
        if flag:
            # print ("Was not able to get a good score for {}".format(chosen))
            continue
        vf = vi + v
        new_x, new_y = vf[0], vf[1]

        new_loss, new_deficit = delta_loss(coords, wind_inst_freq, chosen, new_x, new_y, original_deficit)

        if new_loss <= old_loss:
            coords[chosen][0], coords[chosen][1] = new_x, new_y
            improvement = old_loss - new_loss
            old_loss = new_loss
            original_deficit = new_deficit
            print()
            print("Total iter num: {}".format(iteration))
            print("current average : {}".format(old_loss))
            print("Chose windmill {} and got an improvement of {} units in the average AEP".format(chosen, improvement))
            std_dis += INC_SIZE
        else:
            # print("Chose windmill {} and did not get an improvement".format(chosen))
            if std_dis - INC_SIZE > INIT_STD_DIS: 
                std_dis -= INC_SIZE