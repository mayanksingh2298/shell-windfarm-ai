#andha crude local search
import numpy as np
from evaluate import checkConstraints, binWindResourceData, getAEP, loadPowerCurve, getTurbLoc, preProcessing
from datetime import datetime
import random
from args import make_args
from tqdm import tqdm
from utils import score, initialise_valid, initialise_periphery, min_dist_from_rest, delta_score
from constants import *
from joblib import Parallel, delayed, parallel_backend
import joblib
import time
import math
from operator import itemgetter
from heapq import heappush, heappop
import sys

args = make_args()

EPSILON = args.step
RANDOM_RESTART_THRESH = 1000000000
# RANDOM_EPS = args.random_eps
RANDOM_EPS = True
# DIRECTIONS = args.directions
DIRECTIONS = 36
IMPROVEMENT_THRESH = 0.0001
GREEDY_TURBINE_PROB = 0.5
PARALLEL_JOBS = 8
DELTA = (2*np.pi)/DIRECTIONS
STEP = 20
# if RANDOM_EPS:
    # RANDOM_RESTART_THRESH = 10*RANDOM_RESTART_THRESH
    # LOWER_LIM = 50
    # UPPER_LIM = 500
LOWER_LIM = 50
UPPER_LIM = 3950

def save_csv(coords, filename=None):
    # f = open("submissions/temp_{}_avg_aep_{}m_jump_{}_directions_{}_iterations_{}.csv"
        # .format(score(coords, wind_inst_freq), EPSILON,DIRECTIONS,iteration, str(datetime.now()).replace(':','')), "w")
    if filename is None:
        f = open("submissions/ati.csv",'w')
    else:
        f = open(filename,'w')

    np.savetxt(f, coords, delimiter=',', header='x,y', comments='', fmt='%1.8f')
    f.close()

def optimise_turbine(option, coords, x, y, chosen, old_score, original_deficit):
    global DELTA, LOWER_LIM, UPPER_LIM, PRECISION, IMPROVEMENT_THRESH
    angle = option*DELTA
    improvement = 0
    coords = coords.copy()
    new_coords = coords.copy()

    # Finding equation of the line of direction
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
        if dist[i] < 400:
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
                segments.append((max(LOWER_LIM, midx-distx-PRECISION), min(UPPER_LIM, midx+distx+PRECISION)))
            else:
                segments.append((max(LOWER_LIM, midy-disty-PRECISION), min(UPPER_LIM, midy+disty+PRECISION)))

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
                traverse = np.concatenate([traverse, np.arange(last_segment, upper_lim[key]+PRECISION, mystep)])
        if segments[i][1] > last_segment:
            last_segment = segments[i][1]
    # print("traverse: ", traverse)
    if traverse.shape[0] > 0:
        if traverse[-1] <= (upper_lim[key]-mystep):
            traverse = np.concatenate([traverse, np.arange(last_segment, upper_lim[key]+PRECISION, mystep)])
    elif not segments_not_empty:
        # This means traverse is empty because of no turbines intersecting
        traverse = np.concatenate([traverse, np.arange(lower_lim[key], upper_lim[key]+PRECISION, mystep)])

    if angle >= (np.pi - PRECISION):
        traverse = traverse[::-1]

    # print("STARTS", option, traverse.shape)
    # if traverse.shape[0] > 200:
    #   import ipdb
    #   ipdb.set_trace()
    # print(traverse)
    # for EPSILON in range(LOWER_LIM, UPPER_LIM+1, 10):
    for ele in traverse:
        if m is not None:
            new_x = ele
            new_y = m*new_x + intercept
        else:
            new_y = ele
            new_x = intercept

        #check coords yahi pe
        if not (50 < new_x < 3950 and 50 < new_y < 3950):
            continue
        
        # #just need to check the latest point
        # #also check dist from nbrs of this new pt
        # min_d = min_dist_from_rest(chosen, coords, new_x, new_y)
        # if min_d <= 400 + PRECISION:
            # continue

        copied = coords.copy() # an undo can be done to optimise later
        # copied = coords
        # tmp_x, tmp_y = copied[chosen][0], copied[chosen][1]
        copied[chosen][0], copied[chosen][1] = new_x, new_y
        try:
            new_score, _ = delta_score(coords, wind_inst_freq, chosen, new_x, new_y, original_deficit)
            # new_score = score(copied, wind_inst_freq)
        except:
            import ipdb
            ipdb.set_trace()
            # return -1, None
        new_improvement = new_score - old_score
        if new_improvement > improvement:
            improvement = new_improvement
            new_coords = copied.copy()
            # copied[chosen][0], copied[chosen][1] = tmp_x, tmp_y
            break
        # copied[chosen][0], copied[chosen][1] = tmp_x, tmp_y

    if improvement < IMPROVEMENT_THRESH:
        return -1, None
    else:
        return improvement, new_coords

if __name__ == "__main__":
    
    #initialise coords with working values
    # years = [2007]
    years = [2007, 2008, 2009, 2013, 2014, 2015, 2017]
    year_wise_dist = np.array([binWindResourceData('../../data/WindData/wind_data_{}.csv'.format(year)) for year in years])
    wind_inst_freq = np.mean(year_wise_dist, axis = 0) #over all years

    if len(args.load) > 0:
        coords = getTurbLoc(args.load) #supposed to be a numpy array of shape 50 x 2
    else:
        coords = initialise_periphery()
    #to check initialiser

    iteration = -1
    # coords = initialise_valid()
    # coords = getTurbLoc("submissions/best_yet-528.95.csv")

    total_iterations = 0
    num_restarts = 0
    iters_with_no_inc = 0

    time_avg = []
    # Don't select latest turbine for atleast 10 chances
    # last_turbine = []
    waste_queue = []
    working_queue = []
    global_improvement = 0
    old_score,turbine_powers,original_deficit = score(coords,wind_inst_freq, to_print=False, with_deficit=True, get_each_turbine_power=True) 
    # print(turbine_powers.shape)
    for ii in range(turbine_powers.shape[0]):
        # print(turbine_powers[ii])
        heappush(working_queue, (turbine_powers[ii][0], ii))

    best_coords = None
    best_score = 0
    for ii in tqdm(range(9999999999999)):
        if iters_with_no_inc >= RANDOM_RESTART_THRESH:
            save_csv(coords)
            iters_with_no_inc = 0
            iteration = 0
            num_restarts += 1
            coords = initialise_valid()


        # if RANDOM_EPS:
        #   EPSILON = np.random.uniform(LOWER_LIM, UPPER_LIM)
        print("Waste_queue: ", waste_queue)
        print("Working_queue: ", [wq[1] for wq in working_queue])
        print("global_improvement: ", global_improvement)
        total_iterations += 1
        iteration += 1

        old_score,turbine_powers,original_deficit = score(coords,wind_inst_freq, to_print=False, with_deficit=True, get_each_turbine_power=True) 

        if len(working_queue) == 0:
            if global_improvement == 0:
                print("I guess maximised to the extent possible using just one turbine")
                save_csv(coords, "submissions/ati_"+str(ii)+"_"+str(best_score)+".csv")
                coords = initialise_valid()
                num_restarts += 1
                old_score,turbine_powers,original_deficit = score(coords,wind_inst_freq, to_print=False, with_deficit=True, get_each_turbine_power=True) 
                for kk in range(turbine_powers.shape[0]):
                    heappush(working_queue, (turbine_powers[kk][0], kk))
                waste_queue = []
            else:
                for kk in range(len(waste_queue)):
                    heappush(working_queue, waste_queue[kk])
                waste_queue = []
                global_improvement = 0
        
        chosen = -1
        # maxi = 0
        # while((chosen < 0) or (chosen in last_turbine)):
            #select turbine to move
        if(random.random()<=GREEDY_TURBINE_PROB):
            chosen_turb = heappop(working_queue)
            chosen = chosen_turb[1]
        else:
            index = np.random.randint(0,len(working_queue)) #50 is not included
            chosen_turb = working_queue.pop(index)
            chosen = chosen_turb[-1]

        #now lets see whether we can improve upon our score
        # coords = coords.copy()
        x, y = coords[chosen]

        start = time.time()
        data = Parallel(n_jobs=PARALLEL_JOBS)(delayed(optimise_turbine)(option,coords,x,y,chosen,old_score,original_deficit) for option in range(DIRECTIONS))
        # for option in range(DIRECTIONS):
        #   angle = option*DELTA
        #   improvement = 0
        #   new_coords = coords
        #   for EPSILON in range(LOWER_LIM, UPPER_LIM+1, 10):
        #       new_x = x + EPSILON*np.cos(angle)
        #       new_y = y + EPSILON*np.sin(angle)


        #       #check coords yahi pe
        #       if not (50 < new_x < 3950 and 50 < new_y < 3950):
        #           continue
        #           #just need to check the latest point
        #       #also check dist from nbrs of this new pt
        #       min_d = min_dist_from_rest(chosen, coords, new_x, new_y)
        #       if min_d <= 400 + PRECISION:
        #           continue

        #       copied = coords.copy() # an undo can be done to optimise later
        #       copied[chosen][0], copied[chosen][1] = new_x, new_y 
        #       new_score = score(copied, wind_inst_freq)
        #       new_improvement = new_score - old_score
        #       if new_improvement > improvement:
        #           improvement = new_improvement
        #           new_coords = copied

        #   # if improvement >= best_improvement:
        #   if improvement >= IMPROVEMENT_THRESH:
        #       best_improvement = improvement
        #       best_coords = new_coords
        #       coords = best_coords
        #       print("new_score:",new_score,"Iteration:",total_iterations)
        #       save_csv(coords)
        #       iters_with_no_inc = 0 
        #       break
        #   else:
        #       iters_with_no_inc += 1
        # print(data)
        # # if improvement >= best_improvement:

        improvement = max(data, key = lambda i : i[0])
        if improvement[1] is not None:
            new_coords = improvement[1].copy()
            print("Improved: ", improvement[0], chosen, coords[chosen], new_coords[chosen])
        else:
            print("Improved: ", improvement[0], chosen, coords[chosen], None)

        improvement = improvement[0]
        new_score = old_score + improvement
        if improvement >= IMPROVEMENT_THRESH:
            best_score = old_score+improvement
            best_coords = new_coords
            coords = best_coords
            print("new_score:",new_score,"Iteration:",total_iterations)
            save_csv(best_coords)
            iters_with_no_inc = 0
            global_improvement += improvement
            heappush(working_queue, chosen_turb)
        else:
            iters_with_no_inc += 1
            waste_queue.append(chosen_turb)
            # break

        time_avg.append(time.time() - start)
        print("Time for one turbine: ", sum(time_avg)/len(time_avg))
