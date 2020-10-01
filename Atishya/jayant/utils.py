# utils.py
from evaluate import checkConstraints, binWindResourceData, getAEP, loadPowerCurve, getTurbLoc, preProcessing, delta_AEP
import numpy as np
from constants import *
import random
import math

def fetch_movable_segments(coords, chosen, angle):
    global LOWER_LIM, UPPER_LIM, PRECISION
    # Finding equation of the line of direction
    x, y = coords[chosen]
    m = np.tan(angle)
    if abs(m) < PRECISION:
        m = 0
        intercept = y
        if angle < np.pi - PRECISION:
            lower_lim = (x,y)
            upper_lim = (UPPER_LIM, y)
        else:
            lower_lim = (LOWER_LIM, y)
            upper_lim = (x,y)
    elif abs(m) > 500:
        m = None
        intercept = x
        if angle < np.pi - PRECISION:
            lower_lim = (x,y)
            upper_lim = (x,UPPER_LIM)
        else:
            lower_lim = (x,LOWER_LIM)
            upper_lim = (x,y)
    else:
        intercept = y - m*x
        denominator = m + 1/m
        # Setting the lower and upper limit of traversal of turbine
        if (angle < (np.pi/2 - PRECISION)):
            lower_lim = (x,y)
            if m*UPPER_LIM + intercept > UPPER_LIM:
                upper_lim = ((UPPER_LIM - intercept)/m - PRECISION, UPPER_LIM)
            else:
                upper_lim = (UPPER_LIM, m*UPPER_LIM + intercept - PRECISION)
        elif (angle > (3*np.pi/2 + PRECISION)):
            lower_lim = (x,y)
            if m*UPPER_LIM + intercept < LOWER_LIM:
                upper_lim = ((LOWER_LIM - intercept)/m - PRECISION, LOWER_LIM)
            else:
                upper_lim = (UPPER_LIM, m*UPPER_LIM + intercept + PRECISION)            
        elif (angle > (np.pi/2 + PRECISION)) and (angle < (np.pi - PRECISION)):
            if m*LOWER_LIM + intercept > UPPER_LIM:
                lower_lim = ((UPPER_LIM - intercept)/m + PRECISION, UPPER_LIM)
            else:
                lower_lim = (LOWER_LIM, m*LOWER_LIM + intercept - PRECISION)
            upper_lim = (x,y)
        else:
            if m*LOWER_LIM + intercept < LOWER_LIM:
                lower_lim = ((LOWER_LIM - intercept)/m + PRECISION, LOWER_LIM)
            else:
                lower_lim = (LOWER_LIM, m*LOWER_LIM + intercept + PRECISION)
            upper_lim = (x,y)

    if m is not None:
        dist = np.abs(coords[:,1] - m*coords[:,0] - intercept) / math.sqrt(1+m*m)
    else:
        dist = np.abs(coords[:,0] - intercept)
    segments = []
    # print(dist.shape)
    for i in range(50):
        if dist[i] < 400 and i != chosen:
            if m is not None and m != 0:
                cp = coords[i,1] + ((1/m) * coords[i,0])
                midx = (cp - intercept)/denominator
                midy = (m*cp + (intercept/m))/denominator
                loc_dist = math.sqrt(400*400 - dist[i]*dist[i])
                distx = abs(loc_dist*np.cos(angle))
                disty = abs(loc_dist*np.sin(angle))
            elif m == 0:
                midx = coords[i,0]
                midy = intercept
                loc_dist = math.sqrt(400*400 - dist[i]*dist[i])
                distx = loc_dist
                disty = 0
            else:
                midx = intercept
                midy = coords[i,1]
                loc_dist = math.sqrt(400*400 - dist[i]*dist[i])
                distx = 0
                disty = loc_dist

            # If slope is not infinity, we'll travel on x axis, else on y axis
            if m is not None:
                above = min(UPPER_LIM, midx+distx+PRECISION)
                below = max(LOWER_LIM, midx-distx-PRECISION)
                if above > below:
                    segments.append((below, above))
            else:
                above = min(UPPER_LIM, midy+disty+PRECISION)
                below = max(LOWER_LIM, midy-disty-PRECISION)
                if above > below:
                    segments.append((below, above))

            # if m is not None and m > 0:
            #   segments.append((midx-distx-PRECISION, midy-disty-PRECISION), (midx+distx+PRECISION, midy+disty+PRECISION))
            # else:
            #   segments.append((midx-distx-PRECISION, midy+disty+PRECISION), (midx+distx+PRECISION, midy-disty-PRECISION))
    if m is not None:
        key = 0
    else:
        key = 1
    segments.sort()
    last_segment = lower_lim[key]
    traverse = np.empty(0)
    if m is not None:
        mystep = abs(STEP*np.cos(angle))
    else:
        mystep = STEP

    lasti = 0
    segments_not_empty = False
    for i in range(len(segments)):
        # If my current segment is in between
        if (segments[i][0] < last_segment) and (segments[i][1] > last_segment):
            segments_not_empty = True
            last_segment = segments[i][1]
            lasti = i
            break
        elif (segments[i][0] >= last_segment) and (segments[i][1] > last_segment):
            lasti = i-1
            break

    for i in range(lasti+1, len(segments)):
        segments_not_empty = True
        if (segments[i][0] - last_segment >= mystep):
            if (segments[i][0] < upper_lim[key]): 
                traverse = np.concatenate([traverse, np.arange(last_segment, segments[i][0]+PRECISION, mystep)])
            else:
                if last_segment < upper_lim[key]:
                    traverse = np.concatenate([traverse, np.arange(last_segment, upper_lim[key]-PRECISION, mystep)])
                break
        if segments[i][1] > last_segment:
            last_segment = segments[i][1]
    # print("traverse: ", traverse)
    if traverse.shape[0] > 0:
        if traverse[-1] <= (upper_lim[key]-mystep):
            if last_segment < upper_lim[key]-PRECISION:
                traverse = np.concatenate([traverse, np.arange(last_segment, upper_lim[key]-PRECISION, mystep)])
    elif not segments_not_empty:
        # This means traverse is empty because of no turbines intersecting
        traverse = np.concatenate([traverse, np.arange(lower_lim[key], upper_lim[key]-PRECISION, mystep)])
    else:
        if last_segment <= upper_lim[key]-PRECISION:
            traverse = np.concatenate([traverse, np.arange(last_segment, upper_lim[key]-PRECISION, mystep)])

    if angle >= (np.pi - PRECISION):
        traverse = traverse[::-1]

    return traverse, m, intercept

def initialise_valid():
    #for now return the same everytime to compare methods etc
    # return getTurbLoc('../data/turbine_loc_test.csv')
    # this can be done quicker with matrix operations
    # 50 <= x,y <= 3950
    # define an 9x9 grid, choose 50 out of 81 boxes to fill
    # choosing pts in 100, 3800 to give them room on the boundaries as well

    # jump = 3800/9 # >400
    # region = 3800/9 - 400
    region = (3900 - 400*8)/9
    # region = (3800 - 400*8)/9
    jump = 400 + region

    data = []
    # generate 81 points then choose just 50
    for i in range(9):
        # s_x = 100 + (jump)*i
        s_x = 50 + (jump)*i
        for j in range(9):
            s_y = 50 + jump*j
            
            x = s_x + np.random.uniform(0, region)
            y = s_y + np.random.uniform(0, region)

            data.append((x,y))

    random.shuffle(data)
    return np.array(data[:50])

def initialise_periphery():
    data = []
    delta = (3900 - 2*300)/8

    #36 pts on the periphery
    for i in range(9):
        data.append((50 + 300 + delta*i,50 + 1) )
        data.append((50 + 300 + delta*i,3950 - 1) )
        data.append((50 +1, 50 + 300 + delta*i) )
        data.append((3950 - 1,50 + 300 + delta*i) )

    #now fit 16 points in the centre
    # margin = 402 #atleast itna, and less than 1200 ish
    margin = 1000
    inner_dis = (3900 - margin*2)/3
    for i in range(4):
        for j in range(4):
            x = 50 + margin + i*inner_dis
            y = 50 + margin + j*inner_dis
            data.append((x,y)) 

    # random.shuffle(data)
    return np.array(data[:50])


def score(coords, wind_inst_freq, to_print = False, with_deficit = False, get_each_turbine_power=False):
    # success = checkConstraints(coords, turb_diam)
    # if not success:
    #   print("CONSTRAINTS VIOLATED")
    #   raise "VIOLATION"
    #   return MINIMUM 
    ret = getAEP(turb_rad, coords, power_curve, wind_inst_freq, 
        n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t, get_each_turbine_power=get_each_turbine_power, with_deficit=with_deficit) 

    if to_print:
        if with_deficit or get_each_turbine_power:
            AEP, turbine_powers, deficit = ret
        else:
            AEP = ret
        print('Average AEP over train years is {}', "%.12f"%(AEP), 'GWh')
    return ret

def delta_score(coords, wind_inst_freq, chosen, new_x, new_y, original_deficit):
    return delta_AEP(turb_rad, coords, power_curve, wind_inst_freq, 
                n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t,
                chosen, new_x, new_y, original_deficit)

def min_dist_from_rest(chosen, coords, new_x, new_y):
    min_d = min([(new_x - coords[i][0])**2 + (new_y - coords[i][1])**2 if i!=chosen else MAXIMUM for i in range(50)])
    min_d = min_d**0.5
    return min_d