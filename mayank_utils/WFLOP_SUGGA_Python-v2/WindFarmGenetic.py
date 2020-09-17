import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from tqdm import tqdm
from datetime import datetime
from Farm_Evaluator_Vec import *
import random
import copy

__version__ = "1.0.0"


class WindFarmGenetic:
    elite_rate = 0.2 # elite rate: parameter for genetic algorithm
    cross_rate = 0.6 # crossover rate: parameter for genetic algorithm
    random_rate = 0.5 # random rate: parameter for genetic algorithm
    mutate_rate = 0.1 # mutation rate: parameter for genetic algorithm
    turbine = None
    pop_size = 0 # population size : how many individuals in a population
    N = 0 # number of wind turbines
    rows = 0 # how many cell rows the wind farm are divided into
    cols = 0 # how many colus the wind farm land are divided into
    iteration = 0 # how many iterations the genetic algorithm run
    NA_loc=None # not available, not usable locations index list (the index starts from 1)
    cell_width = 0  # cell width
    cell_width_half = 0  # half cell width


    
    

    # constructor of the class
    def __init__(self, rows=21, cols=21, N=0,NA_loc=None, pop_size=100, iteration=200,cell_width=0, elite_rate=0.2,
                 cross_rate=0.6, random_rate=0.5, mutate_rate=0.1):
        self.turbine = GE_1_5_sleTurbine()
        self.rows = rows
        self.cols = cols
        self.N = N
        self.pop_size = pop_size
        self.iteration = iteration

        self.cell_width = cell_width
        self.cell_width_half = cell_width * 0.5

        self.elite_rate = elite_rate
        self.cross_rate = cross_rate
        self.random_rate = random_rate
        self.mutate_rate = mutate_rate

        self.init_pop = None
        self.init_pop_NA = None
        self.init_pop_nonezero_indices = None
        self.NA_loc=NA_loc

        # Custom 
        self.turb_rad = 50
        years = [2007,2008,2009,2013,2014,2015,2017]
        year_wise_dist = [binWindResourceData("../../data/WindData/wind_data_{}.csv".format(year)) for year in years]
        self.wind_inst_freq = np.mean(year_wise_dist, axis = 0)

        self.power_curve = loadPowerCurve('../../data/power_curve.csv')
        # self.wind_inst_freq =  binWindResourceData('../../data/WindData/wind_data_2017.csv')
        self.n_wind_instances, self.cos_dir, self.sin_dir, self.wind_sped_stacked, self.C_t = preProcessing(self.power_curve)

        return


    # choose wind distribution : 1 direction, 1 speed
    def init_1_direction_1_N_speed_13(self):
        self.theta = np.array([0], dtype=np.float32)
        self.velocity = np.array([13.0], dtype=np.float32)
        self.f_theta_v = np.array([[1.0]], dtype=np.float32)
        return

    # choose wind distribution : 4 directions, 1 speed
    def init_4_direction_1_speed_13(self):
        self.theta = np.array(
            [0, 3 * np.pi / 6.0, 6 * np.pi / 6.0, 9 * np.pi / 6.0], dtype=np.float32)  # 1.0/4
        self.velocity = np.array([13.0], dtype=np.float32)  # 1
        self.f_theta_v = np.array([[0.25], [0.25], [0.25], [0.25]], dtype=np.float32)
        return

    # choose wind distribution : 6 direction, 1 speed
    def init_6_direction_1_speed_13(self):
        self.theta = np.array([0, np.pi / 3.0, 2 * np.pi / 3.0, 3 * np.pi / 3.0, 4 * np.pi / 3.0, 5 * np.pi / 3.0],
                              dtype=np.float32)  # 0.2, 0,3 0.2  0. 1 0.1 0.1
        self.velocity = np.array([13.0], dtype=np.float32)  # 1
        self.f_theta_v = np.array([[0.2], [0.3], [0.2], [0.1], [0.1], [0.1]], dtype=np.float32)
        return

    def init_custom_windspeed(self,path):
        def binWindResourceData(wind_data_file_name):
            # Load wind data. Then, extracts the 'drct', 'sped' columns
            f = open(wind_data_file_name,'r').readlines()
            l = []
            for line in f[1:]:
                line = line.strip().split(',')
                l.append([])
                l[-1].extend([float(line[1]),float(line[2])])
            wind_resource = np.array(l,dtype = np.float32)
            
            # direction 'slices' in degrees
            slices_drct   = np.roll(np.arange(10, 361, 10, dtype=np.float32), 1)
            ## slices_drct   = [360, 10.0, 20.0.......340, 350]
            n_slices_drct = slices_drct.shape[0]
            
            # speed 'slices'
            slices_sped   = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 
                                18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0]
            n_slices_sped = len(slices_sped)-1

            
            # placeholder for binned wind
            binned_wind = np.zeros((n_slices_drct, n_slices_sped), 
                                   dtype = np.float32)
            
            # 'trap' data points inside the bins. 
            for i in range(n_slices_drct):
                for j in range(n_slices_sped):     
                    
                    # because we already have drct in the multiples of 10
                    foo = wind_resource[(wind_resource[:,0] == slices_drct[i])] 

                    foo = foo[(foo[:,1] >= slices_sped[j]) 
                                  & (foo[:,1] <  slices_sped[j+1])]
                    
                    binned_wind[i,j] = foo.shape[0] 
            
            wind_inst_freq   = binned_wind/np.sum(binned_wind)
            
            return(wind_inst_freq)
        self.theta = np.roll(np.arange(10, 361, 10, dtype=np.float32), 1)
        self.velocity = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 
                        19.0, 21.0, 23.0, 25.0, 27.0, 29.0], dtype=np.float32)
        self.f_theta_v = binWindResourceData(path)

    def custom_single_pop(self):
        # works only for rows = cols = 40
        rows = 40
        while(True):
            indices = set()
            for i in range(10):
                indices.add(1+4*i)
            for i in range(10):
                indices.add(1+160*i)
            for i in range(10):
                indices.add(37+160*i)
            for i in range(10):
                indices.add(1441+4*i)
            indices = list(indices)
            assert len(indices)==36
            newlis = []
            for i in range(8):
                newlis.append(5+4*i)
            random.shuffle(newlis)
            newlis = newlis[:7]
            for i in newlis:
                while True:
                    x = random.randint(1,8)
                    y = random.randint(1,8)
                    if(x!=y):
                        break
                indices.append(i + 160*x)
                indices.append(i + 160*y)
            pop = [0]*(rows*rows)
            for i in indices:
                pop[i-1] = 1
            assert sum(pop)==50
            if(checkConstraints(self.convert_to_realcoords(pop))[0]):
                return pop

    def convert_to_realcoords(self,pop,rows = 40, cols = 40):
        coords = []
        lis = pop

        for i in range(1,rows*cols+1):
            if(lis[i-1]==1):
                x = (i - rows * ((i-1)//rows))*(4000/rows) - (4000/rows)/2
                y = ((i-1)//rows)*(4000/rows) + (4000/rows)/2
                coords.append([x,y])
        coords = np.array(coords)
        return coords

    def custom_fitness(self,pop,pop_size,rows=10,cols=10,N=50,po=None):
        fitness = []
        for i in range(pop_size):
            # coords = []
            # lis = pop[i,:].tolist()
            # for i in range(1,rows*cols+1):
            #     if(lis[i-1]==1):
            #         x = (i - rows * ((i-1)//rows))*(4000/rows) - (4000/rows)/2
            #         y = ((i-1)//rows)*(4000/rows) + (4000/rows)/2
            #         coords.append([x,y])
            # coords = np.array(coords)
            coords = self.convert_to_realcoords(pop[i,:].tolist(),rows,cols)
            # constraints,err = checkConstraints(coords)
            # if(not constraints):
            #     AEP = 0
            # else:
            AEP = getAEP(self.turb_rad, coords, self.power_curve, self.wind_inst_freq, 
                self.n_wind_instances, self.cos_dir, self.sin_dir, self.wind_sped_stacked, self.C_t)
            fitness.append(AEP)
        fitness = np.array(fitness)
        return fitness

    # the total cost
    def cost(self, N):
        return 1.0 * N * (2.0 / 3.0 + 1.0 / 3.0 * math.exp(-0.00174 * N ** 2))

    # generate initial population
    def gen_init_pop_NA(self):
        # self.init_pop,self.init_pop_NA = LayoutGridMCGenerator.gen_pop_with_NA_loc(rows=self.rows, cols=self.cols,NA_loc=self.NA_loc, n=self.pop_size, N=self.N)
        
        self.init_pop = []
        self.init_pop_NA = []
        for i in range(self.pop_size):
            pop = self.custom_single_pop()
            self.init_pop.append(pop)
            self.init_pop_NA.append(pop)
        self.init_pop = np.array(self.init_pop)
        self.init_pop_NA = np.array(self.init_pop_NA)

        self.init_pop_nonezero_indices = np.zeros((self.pop_size, self.N), dtype=np.int32)
        for ind_init_pop in range(self.pop_size):
            ind_indices = 0
            for ind in range(self.rows * self.cols):
                if self.init_pop[ind_init_pop, ind] == 1:
                    self.init_pop_nonezero_indices[ind_init_pop, ind_indices] = ind
                    ind_indices += 1
        return

    # save initial population
    def save_init_pop_NA(self, fname,fname_NA):
        np.savetxt(fname, self.init_pop, fmt='%d', delimiter="  ")
        np.savetxt(fname_NA, self.init_pop_NA, fmt='%d', delimiter="  ")
        return


    # load initial population
    def load_init_pop_NA(self, fname,fname_NA):
        self.init_pop = np.genfromtxt(fname, delimiter="  ", dtype=np.int32)
        self.init_pop_NA = np.genfromtxt(fname_NA, delimiter="  ", dtype=np.int32)
        self.init_pop_nonezero_indices = np.zeros((self.pop_size, self.N), dtype=np.int32)
        for ind_init_pop in range(self.pop_size):
            ind_indices = 0
            for ind in range(self.rows * self.cols):
                if self.init_pop[ind_init_pop, ind] == 1:
                    self.init_pop_nonezero_indices[ind_init_pop, ind_indices] = ind
                    ind_indices += 1
        return

    # calculate total rate power
    def cal_P_rate_total(self):
        f_p = 0.0
        for ind_t in range(len(self.theta)):
            for ind_v in range(len(self.velocity)):
                f_p += self.f_theta_v[ind_t, ind_v]*self.turbine.P_i_X(self.velocity[ind_v])
        return self.N * f_p


    # generate the location index coordinate and average power output at each location index coordinate
    # location index coordinate : in the cells, the cell with index 1 has location index (0,0) and the cell 2 has (1,0)
    # store the location index coordinate in x.dat and average power in y.dat
    def mc_gen_xy_NA(self, rows, cols, layouts, n, N, xfname, yfname):
        layouts_cr = np.zeros((rows * cols, 2), dtype=np.int32)  # layouts column row index
        n_copies = np.sum(layouts, axis=0)
        layouts_power = np.zeros((n, rows * cols), dtype=np.float32)
        self.mc_fitness(pop=layouts, rows=rows, cols=cols, pop_size=n, N=N, lp=layouts_power)
        sum_layout_power = np.sum(layouts_power, axis=0)
        mean_power = np.zeros(rows * cols, dtype=np.float32)
        for i in range(rows * cols):
            if n_copies[i]>0:
                mean_power[i] = sum_layout_power[i] / n_copies[i]
        for ind in range(rows * cols):
            r_i = np.floor(ind / cols)
            c_i = np.floor(ind - r_i * cols)
            layouts_cr[ind, 0] = c_i
            layouts_cr[ind, 1] = r_i
        np.savetxt(xfname, layouts_cr, fmt='%d', delimiter="  ")
        np.savetxt(yfname, mean_power, fmt='%f', delimiter="  ")
        return

    # calculate fitness value of the population
    def mc_fitness(self, pop, rows, cols, pop_size, N, lp):
        for i in range(pop_size):
            print("layout {}...".format(i))
            xy_position = np.zeros((2, N), dtype=np.float32)  # x y position
            cr_position = np.zeros((2, N), dtype=np.int32)  # column row position
            ind_position = np.zeros(N, dtype=np.int32)
            ind_pos = 0
            for ind in range(rows * cols):
                if pop[i, ind] == 1:
                    r_i = np.floor(ind / cols)
                    c_i = np.floor(ind - r_i * cols)
                    cr_position[0, ind_pos] = c_i
                    cr_position[1, ind_pos] = r_i
                    xy_position[0, ind_pos] = c_i * self.cell_width + self.cell_width_half
                    xy_position[1, ind_pos] = r_i * self.cell_width + self.cell_width_half
                    ind_position[ind_pos] = ind
                    ind_pos += 1
            lp_power_accum = np.zeros(N, dtype=np.float32)  # a specific layout power accumulate
            for ind_t in range(len(self.theta)):
                for ind_v in range(len(self.velocity)):
                    trans_matrix = np.array(
                        [[np.cos(self.theta[ind_t]), -np.sin(self.theta[ind_t])],
                         [np.sin(self.theta[ind_t]), np.cos(self.theta[ind_t])]],
                        np.float32)
                    trans_xy_position = np.matmul(trans_matrix, xy_position)
                    speed_deficiency = self.wake_calculate(trans_xy_position, N)

                    actual_velocity = (1 - speed_deficiency) * self.velocity[ind_v]
                    lp_power = self.layout_power(actual_velocity,
                                                     N)  # total power of a specific layout specific wind speed specific theta
                    lp_power = lp_power * self.f_theta_v[ind_t, ind_v]
                    lp_power_accum += lp_power

            lp[i, ind_position] = lp_power_accum

        return


    # calculate wake effect
    def wake_calculate(self, trans_xy_position, N):
        sorted_index = np.argsort(-trans_xy_position[1, :])  # y value descending
        wake_deficiency = np.zeros(N, dtype=np.float32)
        wake_deficiency[sorted_index[0]] = 0
        for i in range(1, N):
            for j in range(i):
                xdis = np.absolute(trans_xy_position[0, sorted_index[i]] - trans_xy_position[0, sorted_index[j]])
                ydis = np.absolute(trans_xy_position[1, sorted_index[i]] - trans_xy_position[1, sorted_index[j]])
                d = self.cal_deficiency(dx=xdis, dy=ydis, r=self.turbine.rator_radius,
                                        ec=self.turbine.entrainment_const)
                wake_deficiency[sorted_index[i]] += d ** 2

            wake_deficiency[sorted_index[i]] = np.sqrt(wake_deficiency[sorted_index[i]])
        return wake_deficiency

    # ec : entrainment_const
    def cal_deficiency(self, dx, dy, r, ec):
        if dy == 0:
            return 0
        R = r + ec * dy
        inter_area = self.cal_interaction_area(dx=dx, dy=dy, r=r, R=R)
        d = 2.0 / 3.0 * (r ** 2) / (R ** 2) * inter_area / (np.pi * r ** 2)
        return d

    #calculate ineraction area
    def cal_interaction_area(self, dx, dy, r, R):
        if dx >= r + R:
            return 0
        elif dx >= np.sqrt(R ** 2 - r ** 2):
            alpha = np.arccos((R ** 2 + dx ** 2 - r ** 2) / (2 * R * dx))
            beta = np.arccos((r ** 2 + dx ** 2 - R ** 2) / (2 * r * dx))
            A1 = alpha * R ** 2
            A2 = beta * r ** 2
            A3 = R * dx * np.sin(alpha)
            return A1 + A2 - A3
        elif dx >= R - r:
            alpha = np.arccos((R ** 2 + dx ** 2 - r ** 2) / (2 * R * dx))
            beta = np.pi - np.arccos((r ** 2 + dx ** 2 - R ** 2) / (2 * r * dx))
            A1 = alpha * R ** 2
            A2 = beta * r ** 2
            A3 = R * dx * np.sin(alpha)
            return np.pi * r ** 2 - (A2 + A3 - A1)
        else:
            return np.pi * r ** 2

    def layout_power(self, velocity, N):
        power = np.zeros(N, dtype=np.float32)
        for i in range(N):
            power[i] = self.turbine.P_i_X(velocity[i])
        return power

    # conventional genetic algorithm
    def conventional_genetic_alg(self, ind_time=0,result_folder=None):  # conventional genetic algorithm
        # P_rate_total = self.cal_P_rate_total()
        start_time = datetime.now()
        print("conventional genetic algorithm starts....")
        fitness_generations = np.zeros(self.iteration, dtype=np.float32)  # best fitness value in each generation
        best_layout_generations = np.zeros((self.iteration, self.rows * self.cols),
                                           dtype=np.int32)  # best layout in each generation
        best_layout_NA_generations = np.zeros((self.iteration, self.rows * self.cols),
                                           dtype=np.int32)  # best layout in each generation
        power_order = np.zeros((self.pop_size, self.N),
                               dtype=np.int32)  # each row is a layout cell indices. in each layout, order turbine power from least to largest
        pop = np.copy(self.init_pop)
        pop_NA = np.copy(self.init_pop_NA)
        pop_indices = np.copy(self.init_pop_nonezero_indices)  # each row is a layout cell indices.

        eN = int(np.floor(self.pop_size * self.elite_rate))  # elite number
        rN = int(int(np.floor(self.pop_size * self.mutate_rate)) / eN) * eN  # reproduce number
        mN = rN  # mutation number
        cN = self.pop_size - eN - mN  # crossover number

        for gen in tqdm(range(self.iteration)):
            # print("generation {}...".format(gen))
            time1 = time.time()
            fitness_value = self.custom_fitness(pop=pop, rows=self.rows, cols=self.cols, pop_size=self.pop_size,
                                                      N=self.N,
                                                      po=power_order)
            time2 = time.time()
            # fitness_value = self.conventional_fitness(pop=pop, rows=self.rows, cols=self.cols, pop_size=self.pop_size,
                                                      # N=self.N,
                                                      # po=power_order)

            sorted_index = np.argsort(-fitness_value)  # fitness value descending from largest to least

            pop = pop[sorted_index, :]
            pop_NA=pop_NA[sorted_index, :]

            power_order = power_order[sorted_index, :]
            pop_indices = pop_indices[sorted_index, :]

            if gen == 0:
                fitness_generations[gen] = fitness_value[sorted_index[0]]
                best_layout_generations[gen, :] = pop[0, :]
                best_layout_NA_generations[gen, :] = pop_NA[0, :]

                best_pop = copy.deepcopy(pop)
                best_pop_NA = copy.deepcopy(pop_NA)
                best_pop_indices = copy.deepcopy(pop_indices)

            else:
                if fitness_value[sorted_index[0]] > fitness_generations[gen - 1]:
                    fitness_generations[gen] = fitness_value[sorted_index[0]]
                    best_layout_generations[gen, :] = pop[0, :]
                    best_layout_NA_generations[gen, :] = pop_NA[0, :]
                    
                    best_pop = copy.deepcopy(pop)
                    best_pop_NA = copy.deepcopy(pop_NA)
                    best_pop_indices = copy.deepcopy(pop_indices)

                    print("new best val is:",fitness_generations[gen],"at gen:",gen)
                    filename = "{}/conventional_best_layouts_avg.dat".format(result_folder)
                    tosave = best_layout_generations[gen,:].tolist()
                    f = open(filename,'w')
                    for i in tosave:
                        f.write(str(i)+" ")
                    f.close()
                else:
                    fitness_generations[gen] = fitness_generations[gen - 1]
                    best_layout_generations[gen, :] = best_layout_generations[gen - 1, :]
                    best_layout_NA_generations[gen, :] = best_layout_NA_generations[gen - 1, :]

                    pop = copy.deepcopy(best_pop)
                    pop_NA = copy.deepcopy(best_pop_NA)
                    pop_indices = copy.deepcopy(best_pop_indices)

            n_parents, parent_layouts,parent_layouts_NA, parent_pop_indices = self.conventional_select(pop=pop,pop_NA=pop_NA, pop_indices=pop_indices,
                                                                                     pop_size=self.pop_size,
                                                                                     elite_rate=self.elite_rate,
                                                                                     random_rate=self.random_rate)
            time1 = time.time()
            self.conventional_crossover(N=self.N, pop=pop,pop_NA=pop_NA, pop_indices=pop_indices, pop_size=self.pop_size,
                                        n_parents=n_parents,
                                        parent_layouts=parent_layouts,parent_layouts_NA=parent_layouts_NA, parent_pop_indices=parent_pop_indices)
            time1 = time.time()
            self.conventional_mutation(rows=self.rows, cols=self.cols, N=self.N, pop=pop,pop_NA= pop_NA,pop_indices=pop_indices,
                                       pop_size=self.pop_size,
                                       mutation_rate=self.mutate_rate)

            # if(gen%50==0):
            #     end_time = datetime.now()
            #     run_time = (end_time - start_time).total_seconds()
            #     eta_generations = np.copy(fitness_generations)
            #     # eta_generations = eta_generations * (1.0 / P_rate_total)
            #     time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
            #     # filename = "{}/conventional_eta.dat".format(result_folder)
            #     # # np.savetxt(filename, eta_generations, fmt='%f', delimiter="  ")
            #     # f = open(filename,"a")
            #     # f.write(str(eta_generations[gen])+"\n")
            #     # f.close()
            #     filename = "{}/conventional_best_layouts.dat".format(result_folder)
            #     # np.savetxt(filename, best_layout_generations[gen], fmt='%d', delimiter="  ")
            #     tosave = best_layout_generations[gen,:].tolist()
            #     f = open(filename,'w')
            #     for i in tosave:
            #         f.write(str(i)+" ")
            #     f.close()
                # filename = "{}/conventional_best_layouts_NA_N{}_{}_{}.dat".format(result_folder,self.N, ind_time, time_stamp)
                # np.savetxt(filename, best_layout_NA_generations, fmt='%d', delimiter="  ")
                # print("conventional genetic algorithm ends.")
                # filename = "{}/conventional_runtime.txt".format(result_folder)
                # f = open(filename, "a+")
                # f.write("{}\n".format(run_time))
                # f.close()

                # filename = "{}/conventional_eta.txt".format(result_folder)
                # f = open(filename, "a+")
                # f.write("{}\n".format(eta_generations[self.iteration - 1]))
                # f.close()
        return run_time, eta_generations[self.iteration - 1]

    def conventional_mutation(self, rows, cols, N, pop,pop_NA, pop_indices, pop_size, mutation_rate):
        np.random.seed(seed=int(time.time()))
        for i in range(pop_size):
            if np.random.randn() > mutation_rate:
                continue
            while True:
                turbine_pos = np.random.randint(0, cols * rows)
                if pop_NA[i, turbine_pos] == 1:
                    break
            while True:
                null_turbine_pos = np.random.randint(0, cols * rows)
                if pop_NA[i, null_turbine_pos] == 0:
                    break
            pop[i, turbine_pos] = 0
            pop[i, null_turbine_pos] = 1

            pop_NA[i, turbine_pos] = 0
            pop_NA[i, null_turbine_pos] = 1

            for j in range(N):
                if pop_indices[i, j] == turbine_pos:
                    pop_indices[i, j] = null_turbine_pos
                    break
            pop_indices[i, :] = np.sort(pop_indices[i, :])
        return

    def conventional_crossover(self, N, pop,pop_NA, pop_indices, pop_size, n_parents,
                               parent_layouts,parent_layouts_NA, parent_pop_indices):
        n_counter = 0
        np.random.seed(seed=int(time.time()))  # init random seed
        while n_counter < pop_size:
            male = np.random.randint(0, n_parents)
            female = np.random.randint(0, n_parents)
            if male != female:
                cross_point = np.random.randint(1, N)
                if parent_pop_indices[male, cross_point - 1] < parent_pop_indices[female, cross_point]:
                    pop[n_counter, :] = 0
                    pop[n_counter, :parent_pop_indices[male, cross_point - 1] + 1] = parent_layouts[male,
                                                                                     :parent_pop_indices[
                                                                                          male, cross_point - 1] + 1]
                    pop[n_counter, parent_pop_indices[female, cross_point]:] = parent_layouts[female,
                                                                               parent_pop_indices[female, cross_point]:]

                    pop_NA[n_counter, :] = pop[n_counter, :]
                    for i in self.NA_loc:
                        pop_NA[n_counter,i-1]=2
                    pop_indices[n_counter, :cross_point] = parent_pop_indices[male, :cross_point]
                    pop_indices[n_counter, cross_point:] = parent_pop_indices[female, cross_point:]
                    n_counter += 1
        return

    def conventional_select(self, pop,pop_NA, pop_indices, pop_size, elite_rate, random_rate):
        n_elite = int(pop_size * elite_rate)
        parents_ind = [i for i in range(n_elite)]
        np.random.seed(seed=int(time.time()))  # init random seed
        for i in range(n_elite, pop_size):
            if np.random.randn() < random_rate:
                parents_ind.append(i)
        parent_layouts = pop[parents_ind, :]
        parent_layouts_NA = pop_NA[parents_ind, :]
        parent_pop_indices = pop_indices[parents_ind, :]
        return len(parent_pop_indices), parent_layouts,parent_layouts_NA, parent_pop_indices

    def conventional_fitness(self, pop, rows, cols, pop_size, N, po):
        fitness_val = np.zeros(pop_size, dtype=np.float32)
        for i in range(pop_size):

            # layout = np.reshape(pop[i, :], newshape=(rows, cols))
            xy_position = np.zeros((2, N), dtype=np.float32)  # x y position
            cr_position = np.zeros((2, N), dtype=np.int32)  # column row position
            ind_position = np.zeros(N, dtype=np.int32)
            ind_pos = 0
            for ind in range(rows * cols):
                if pop[i, ind] == 1:
                    r_i = np.floor(ind / cols)
                    c_i = np.floor(ind - r_i * cols)
                    cr_position[0, ind_pos] = c_i
                    cr_position[1, ind_pos] = r_i
                    xy_position[0, ind_pos] = c_i * self.cell_width + self.cell_width_half
                    xy_position[1, ind_pos] = r_i * self.cell_width + self.cell_width_half
                    ind_position[ind_pos] = ind
                    ind_pos += 1
            lp_power_accum = np.zeros(N, dtype=np.float32)  # a specific layout power accumulate
            for ind_t in range(len(self.theta)):
                for ind_v in range(len(self.velocity)):
                    trans_matrix = np.array(
                        [[np.cos(self.theta[ind_t]), -np.sin(self.theta[ind_t])],
                         [np.sin(self.theta[ind_t]), np.cos(self.theta[ind_t])]],
                        np.float32)

                    trans_xy_position = np.matmul(trans_matrix, xy_position)
                    speed_deficiency = self.wake_calculate(trans_xy_position, N)

                    actual_velocity = (1 - speed_deficiency) * self.velocity[ind_v]
                    lp_power = self.layout_power(actual_velocity,
                                                     N)  # total power of a specific layout specific wind speed specific theta
                    lp_power = lp_power * self.f_theta_v[ind_t, ind_v]
                    lp_power_accum += lp_power
            sorted_index = np.argsort(lp_power_accum)  # power from least to largest
            po[i, :] = ind_position[sorted_index]
            fitness_val[i] = np.sum(lp_power_accum)
        return fitness_val

    # AGA: adaptive genetic algorithm
    def adaptive_genetic_alg(self, ind_time=0,result_folder=None):  # adaptive genetic algorithm
        P_rate_total = self.cal_P_rate_total()
        start_time = datetime.now()
        print("adaptive genetic algorithm starts....")
        fitness_generations = np.zeros(self.iteration, dtype=np.float32)  # best fitness value in each generation
        best_layout_generations = np.zeros((self.iteration, self.rows * self.cols),
                                           dtype=np.int32)  # best layout in each generation
        best_layout_NA_generations = np.zeros((self.iteration, self.rows * self.cols),
                                              dtype=np.int32)  # best layout in each generation

        power_order = np.zeros((self.pop_size, self.N),
                               dtype=np.int32)  # each row is a layout cell indices. in each layout, order turbine power from least to largest
        pop = np.copy(self.init_pop)
        pop_NA = np.copy(self.init_pop_NA)
        pop_indices = np.copy(self.init_pop_nonezero_indices)  # each row is a layout cell indices.

        eN = int(np.floor(self.pop_size * self.elite_rate))  # elite number
        rN = int(int(np.floor(self.pop_size * self.mutate_rate)) / eN) * eN  # reproduce number
        mN = rN  # mutation number
        cN = self.pop_size - eN - mN  # crossover number

        for gen in range(self.iteration):
            print("generation {}...".format(gen))
            fitness_value = self.adaptive_fitness(pop=pop, rows=self.rows, cols=self.cols, pop_size=self.pop_size,
                                                  N=self.N,
                                                  po=power_order)
            sorted_index = np.argsort(-fitness_value)  # fitness value descending from largest to least

            pop = pop[sorted_index, :]
            pop_NA = pop_NA[sorted_index, :]
            power_order = power_order[sorted_index, :]
            pop_indices = pop_indices[sorted_index, :]
            if gen == 0:
                fitness_generations[gen] = fitness_value[sorted_index[0]]
                best_layout_generations[gen, :] = pop[0, :]
                best_layout_NA_generations[gen, :] = pop_NA[0, :]
            else:
                if fitness_value[sorted_index[0]] > fitness_generations[gen - 1]:
                    fitness_generations[gen] = fitness_value[sorted_index[0]]
                    best_layout_generations[gen, :] = pop[0, :]
                    best_layout_NA_generations[gen, :] = pop_NA[0, :]
                else:
                    fitness_generations[gen] = fitness_generations[gen - 1]
                    best_layout_generations[gen, :] = best_layout_generations[gen - 1, :]
                    best_layout_NA_generations[gen, :] = best_layout_NA_generations[gen - 1, :]
            self.adaptive_move_worst(rows=self.rows, cols=self.cols, pop=pop,pop_NA=pop_NA, pop_indices=pop_indices,
                                     pop_size=self.pop_size, power_order=power_order)



            n_parents, parent_layouts,parent_layouts_NA, parent_pop_indices = self.adaptive_select(pop=pop,pop_NA=pop_NA, pop_indices=pop_indices,
                                                                                 pop_size=self.pop_size,
                                                                                 elite_rate=self.elite_rate,
                                                                                 random_rate=self.random_rate)


            self.adaptive_crossover(N=self.N, pop=pop,pop_NA=pop_NA, pop_indices=pop_indices, pop_size=self.pop_size,
                                    n_parents=n_parents,
                                    parent_layouts=parent_layouts,parent_layouts_NA=parent_layouts_NA,  parent_pop_indices=parent_pop_indices)


            self.adaptive_mutation(rows=self.rows, cols=self.cols, N=self.N, pop=pop,pop_NA=pop_NA, pop_indices=pop_indices,
                                   pop_size=self.pop_size,
                                   mutation_rate=self.mutate_rate)


        end_time = datetime.now()
        run_time = (end_time - start_time).total_seconds()
        eta_generations = np.copy(fitness_generations)
        # eta_generations = eta_generations * (1.0 / P_rate_total)
        time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")

        filename = "{}/adaptive_eta_N{}_{}_{}.dat".format(result_folder,self.N, ind_time, time_stamp)
        np.savetxt(filename, eta_generations, fmt='%f', delimiter="  ")
        filename = "{}/adaptive_best_layouts_N{}_{}_{}.dat".format(result_folder,self.N, ind_time, time_stamp)
        np.savetxt(filename, best_layout_generations, fmt='%d', delimiter="  ")
        filename = "{}/adaptive_best_layouts_NA_N{}_{}_{}.dat".format(result_folder,self.N, ind_time, time_stamp)
        np.savetxt(filename, best_layout_NA_generations, fmt='%d', delimiter="  ")
        print("adaptive genetic algorithm ends.")
        filename = "{}/adaptive_runtime.txt".format(result_folder)
        f = open(filename, "a+")
        f.write("{}\n".format(run_time))
        f.close()

        filename = "{}/adaptive_eta.txt".format(result_folder)
        f = open(filename, "a+")
        f.write("{}\n".format(eta_generations[self.iteration - 1]))
        f.close()

        return run_time, eta_generations[self.iteration - 1]

    def adaptive_move_worst(self, rows, cols, pop,pop_NA, pop_indices, pop_size, power_order):
        np.random.seed(seed=int(time.time()))
        for i in range(pop_size):
            turbine_pos = power_order[i, 0]
            while True:
                null_turbine_pos = np.random.randint(0, cols * rows)
                if pop_NA[i, null_turbine_pos] == 0:
                    break
            pop[i, turbine_pos] = 0
            pop[i, null_turbine_pos] = 1
            pop_NA[i, turbine_pos] = 0
            pop_NA[i, null_turbine_pos] = 1
            power_order[i, 0] = null_turbine_pos
            pop_indices[i, :] = np.sort(power_order[i, :])
        return

    def adaptive_mutation(self, rows, cols, N, pop,pop_NA, pop_indices, pop_size, mutation_rate):
        np.random.seed(seed=int(time.time()))
        for i in range(pop_size):
            if np.random.randn() > mutation_rate:
                continue
            while True:
                turbine_pos = np.random.randint(0, cols * rows)
                if pop_NA[i, turbine_pos] == 1:
                    break
            while True:
                null_turbine_pos = np.random.randint(0, cols * rows)
                if pop_NA[i, null_turbine_pos] == 0:
                    break
            pop[i, turbine_pos] = 0
            pop[i, null_turbine_pos] = 1

            pop_NA[i, turbine_pos] = 0
            pop_NA[i, null_turbine_pos] = 1

            for j in range(N):
                if pop_indices[i, j] == turbine_pos:
                    pop_indices[i, j] = null_turbine_pos
                    break
            pop_indices[i, :] = np.sort(pop_indices[i, :])
        return

    def adaptive_crossover(self, N, pop,pop_NA, pop_indices, pop_size, n_parents,
                           parent_layouts,parent_layouts_NA, parent_pop_indices):
        n_counter = 0
        np.random.seed(seed=int(time.time()))  # init random seed
        while n_counter < pop_size:
            male = np.random.randint(0, n_parents)
            female = np.random.randint(0, n_parents)
            if male != female:
                cross_point = np.random.randint(1, N)
                if parent_pop_indices[male, cross_point - 1] < parent_pop_indices[female, cross_point]:
                    pop[n_counter, :] = 0
                    pop[n_counter, :parent_pop_indices[male, cross_point - 1] + 1] = parent_layouts[male,
                                                                                     :parent_pop_indices[
                                                                                          male, cross_point - 1] + 1]
                    pop[n_counter, parent_pop_indices[female, cross_point]:] = parent_layouts[female,
                                                                               parent_pop_indices[female, cross_point]:]

                    pop_NA[n_counter, :] = pop[n_counter, :]
                    for i in self.NA_loc:
                        pop_NA[n_counter, i - 1] = 2

                    pop_indices[n_counter, :cross_point] = parent_pop_indices[male, :cross_point]
                    pop_indices[n_counter, cross_point:] = parent_pop_indices[female, cross_point:]
                    n_counter += 1
        return

    def adaptive_select(self, pop,pop_NA, pop_indices, pop_size, elite_rate, random_rate):
        n_elite = int(pop_size * elite_rate)
        parents_ind = [i for i in range(n_elite)]
        np.random.seed(seed=int(time.time()))  # init random seed
        for i in range(n_elite, pop_size):
            if np.random.randn() < random_rate:
                parents_ind.append(i)
        parent_layouts = pop[parents_ind, :]
        parent_layouts_NA = pop_NA[parents_ind, :]
        parent_pop_indices = pop_indices[parents_ind, :]
        return len(parent_pop_indices), parent_layouts, parent_layouts_NA, parent_pop_indices

    def adaptive_fitness(self, pop, rows, cols, pop_size, N, po):
        fitness_val = np.zeros(pop_size, dtype=np.float32)
        for i in range(pop_size):

            # layout = np.reshape(pop[i, :], newshape=(rows, cols))
            xy_position = np.zeros((2, N), dtype=np.float32)  # x y position
            cr_position = np.zeros((2, N), dtype=np.int32)  # column row position
            ind_position = np.zeros(N, dtype=np.int32)
            ind_pos = 0
            for ind in range(rows * cols):
                if pop[i, ind] == 1:
                    r_i = np.floor(ind / cols)
                    c_i = np.floor(ind - r_i * cols)
                    cr_position[0, ind_pos] = c_i
                    cr_position[1, ind_pos] = r_i
                    xy_position[0, ind_pos] = c_i * self.cell_width + self.cell_width_half
                    xy_position[1, ind_pos] = r_i * self.cell_width + self.cell_width_half
                    ind_position[ind_pos] = ind
                    ind_pos += 1
            lp_power_accum = np.zeros(N, dtype=np.float32)  # a specific layout power accumulate
            for ind_t in range(len(self.theta)):
                for ind_v in range(len(self.velocity)):
                    trans_matrix = np.array(
                        [[np.cos(self.theta[ind_t]), -np.sin(self.theta[ind_t])],
                         [np.sin(self.theta[ind_t]), np.cos(self.theta[ind_t])]],
                        np.float32)

                    trans_xy_position = np.matmul(trans_matrix, xy_position)
                    speed_deficiency = self.wake_calculate(trans_xy_position, N)

                    actual_velocity = (1 - speed_deficiency) * self.velocity[ind_v]
                    lp_power = self.layout_power(actual_velocity,
                                                     N)  # total power of a specific layout specific wind speed specific theta
                    lp_power = lp_power * self.f_theta_v[ind_t, ind_v]
                    lp_power_accum += lp_power

            sorted_index = np.argsort(lp_power_accum)  # power from least to largest
            po[i, :] = ind_position[sorted_index]

            fitness_val[i] = np.sum(lp_power_accum)

        return fitness_val

    # SUGGA: support vector regression guided genetic algorithm
    def sugga_genetic_alg(self, ind_time=0,svr_model=None,result_folder=None):

        P_rate_total = self.cal_P_rate_total()
        start_time = datetime.now()
        print("Support vector regression guided genetic algorithm starts....")
        fitness_generations = np.zeros(self.iteration, dtype=np.float32)  # best fitness value in each generation
        best_layout_generations = np.zeros((self.iteration, self.rows * self.cols),
                                           dtype=np.int32)  # best layout in each generation
        best_layout_NA_generations = np.zeros((self.iteration, self.rows * self.cols),
                                              dtype=np.int32)  # best layout in each generation

        power_order = np.zeros((self.pop_size, self.N),
                               dtype=np.int32)  # each row is a layout cell indices. in each layout, order turbine power from least to largest
        pop = np.copy(self.init_pop)
        pop_NA = np.copy(self.init_pop_NA)
        pop_indices = np.copy(self.init_pop_nonezero_indices)  # each row is a layout cell indices.

        eN = int(np.floor(self.pop_size * self.elite_rate))  # elite number
        rN = int(int(np.floor(self.pop_size * self.mutate_rate)) / eN) * eN  # reproduce number
        mN = rN  # mutation number
        cN = self.pop_size - eN - mN  # crossover number

        for gen in range(self.iteration):
            print("generation {}...".format(gen))
            fitness_value = self.sugga_fitness(pop=pop, rows=self.rows, cols=self.cols, pop_size=self.pop_size,
                                                        N=self.N,
                                                        po=power_order)
            sorted_index = np.argsort(-fitness_value)  # fitness value descending from largest to least

            pop = pop[sorted_index, :]
            pop_NA = pop_NA[sorted_index, :]
            power_order = power_order[sorted_index, :]
            pop_indices = pop_indices[sorted_index, :]
            if gen == 0:
                fitness_generations[gen] = fitness_value[sorted_index[0]]
                best_layout_generations[gen, :] = pop[0, :]
                best_layout_NA_generations[gen, :] = pop_NA[0, :]
            else:
                if fitness_value[sorted_index[0]] > fitness_generations[gen - 1]:
                    fitness_generations[gen] = fitness_value[sorted_index[0]]
                    best_layout_generations[gen, :] = pop[0, :]
                    best_layout_NA_generations[gen, :] = pop_NA[0, :]
                else:
                    fitness_generations[gen] = fitness_generations[gen - 1]
                    best_layout_generations[gen, :] = best_layout_generations[gen - 1, :]
                    best_layout_NA_generations[gen, :] = best_layout_NA_generations[gen - 1, :]
            self.sugga_move_worst(rows=self.rows, cols=self.cols, pop=pop,pop_NA=pop_NA, pop_indices=pop_indices,
                                           pop_size=self.pop_size, power_order=power_order, svr_model=svr_model)



            n_parents, parent_layouts,parent_layouts_NA,  parent_pop_indices = self.sugga_select(pop=pop,pop_NA=pop_NA, pop_indices=pop_indices,
                                                                                       pop_size=self.pop_size,
                                                                                       elite_rate=self.elite_rate,
                                                                                       random_rate=self.random_rate)


            self.sugga_crossover(N=self.N, pop=pop,pop_NA=pop_NA, pop_indices=pop_indices, pop_size=self.pop_size,
                                          n_parents=n_parents,
                                          parent_layouts=parent_layouts,parent_layouts_NA=parent_layouts_NA, parent_pop_indices=parent_pop_indices)



            self.sugga_mutation(rows=self.rows, cols=self.cols, N=self.N, pop=pop,pop_NA=pop_NA, pop_indices=pop_indices,
                                         pop_size=self.pop_size,
                                         mutation_rate=self.mutate_rate)

        end_time = datetime.now()
        run_time = (end_time - start_time).total_seconds()
        eta_generations = np.copy(fitness_generations)
        eta_generations = eta_generations * (1.0 / P_rate_total)
        time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")

        filename = "{}/sugga_eta_N{}_{}_{}.dat".format(result_folder,self.N, ind_time, time_stamp)
        np.savetxt(filename, eta_generations, fmt='%f', delimiter="  ")
        filename = "{}/sugga_best_layouts_N{}_{}_{}.dat".format(result_folder,self.N, ind_time, time_stamp)
        np.savetxt(filename, best_layout_generations, fmt='%d', delimiter="  ")
        filename = "{}/sugga_best_layouts_NA_N{}_{}_{}.dat".format(result_folder,self.N, ind_time, time_stamp)
        np.savetxt(filename, best_layout_NA_generations, fmt='%d', delimiter="  ")
        print("Support vector regression guided genetic algorithm ends.")
        filename = "{}/sugga_runtime.txt".format(result_folder)
        f = open(filename, "a+")
        f.write("{}\n".format(run_time))
        f.close()
        filename = "{}/sugga_eta.txt".format(result_folder)
        f = open(filename, "a+")
        f.write("{}\n".format(eta_generations[self.iteration - 1]))
        f.close()
        return run_time, eta_generations[self.iteration - 1]

    def sugga_move_worst(self, rows, cols, pop,pop_NA, pop_indices, pop_size, power_order, mars=None,svr_model=None):
        np.random.seed(seed=int(time.time()))
        for i in range(pop_size):
            r = np.random.randn()
            if r < 0.5:
                self.sugga_move_worst_case_random(i=i, rows=rows, cols=cols, pop=pop,pop_NA=pop_NA, pop_indices=pop_indices,
                                                           pop_size=pop_size, power_order=power_order)
            else:
                self.sugga_move_worst_case_best(i=i, rows=rows, cols=cols, pop=pop,pop_NA=pop_NA, pop_indices=pop_indices,
                                                         pop_size=pop_size, power_order=power_order, mars=mars,svr_model=svr_model)

        return

    def sugga_move_worst_case_random(self, i, rows, cols, pop,pop_NA, pop_indices, pop_size, power_order):
        np.random.seed(seed=int(time.time()))
        turbine_pos = power_order[i, 0]
        while True:
            null_turbine_pos = np.random.randint(0, cols * rows)
            if pop_NA[i, null_turbine_pos] == 0:
                break
        pop[i, turbine_pos] = 0
        pop[i, null_turbine_pos] = 1
        pop_NA[i, turbine_pos] = 0
        pop_NA[i, null_turbine_pos] = 1

        power_order[i, 0] = null_turbine_pos
        pop_indices[i, :] = np.sort(power_order[i, :])
        return

    def sugga_move_worst_case_best(self, i, rows, cols, pop,pop_NA, pop_indices, pop_size, power_order, mars,svr_model):
        np.random.seed(seed=int(time.time()))
        n_candiate = 5
        pos_candidate = np.zeros((n_candiate, 2), dtype=np.int32)
        ind_pos_candidate = np.zeros(n_candiate, dtype=np.int32)
        turbine_pos = power_order[i, 0]
        ind_can = 0
        while True:
            null_turbine_pos = np.random.randint(0, cols * rows)
            if pop_NA[i, null_turbine_pos] == 0:
                pos_candidate[ind_can, 1] = int(np.floor(null_turbine_pos / cols))
                pos_candidate[ind_can, 0] = int(np.floor(null_turbine_pos - pos_candidate[ind_can, 1] * cols))
                ind_pos_candidate[ind_can] = null_turbine_pos
                ind_can += 1
                if ind_can == n_candiate:
                    break
        svr_val = svr_model.predict(pos_candidate)
        sorted_index = np.argsort(-svr_val)  # fitness value descending from largest to least
        null_turbine_pos = ind_pos_candidate[sorted_index[0]]

        pop[i, turbine_pos] = 0
        pop[i, null_turbine_pos] = 1

        pop_NA[i, turbine_pos] = 0
        pop_NA[i, null_turbine_pos] = 1

        power_order[i, 0] = null_turbine_pos
        pop_indices[i, :] = np.sort(power_order[i, :])
        return

    def sugga_move_worst_case_worst(self, i, rows, cols, pop, pop_indices, pop_size, power_order, mars):
        np.random.seed(seed=int(time.time()))
        n_candiate = 11
        pos_candidate = np.zeros((n_candiate, 2), dtype=np.int32)
        ind_pos_candidate = np.zeros(n_candiate, dtype=np.int32)
        turbine_pos = power_order[i, 0]
        ind_can = 0
        while True:
            null_turbine_pos = np.random.randint(0, cols * rows)
            if pop[i, null_turbine_pos] == 0:
                pos_candidate[ind_can, 1] = int(np.floor(null_turbine_pos / cols))
                pos_candidate[ind_can, 0] = int(np.floor(null_turbine_pos - pos_candidate[ind_can, 1] * cols))
                ind_pos_candidate[ind_can] = null_turbine_pos
                ind_can += 1
                if ind_can == n_candiate:
                    break
        mars_val = mars.predict(pos_candidate)
        mars_val = mars_val[:, 0]
        sorted_index = np.argsort(mars_val)  # fitness value descending from least to largest
        null_turbine_pos = ind_pos_candidate[sorted_index[0]]
        pop[i, turbine_pos] = 0
        pop[i, null_turbine_pos] = 1
        power_order[i, 0] = null_turbine_pos
        pop_indices[i, :] = np.sort(power_order[i, :])
        return
    # SUGGA move worst
    def sugga_move_worst_case_middle(self, i, rows, cols, pop, pop_indices, pop_size, power_order, mars):
        np.random.seed(seed=int(time.time()))
        n_candiate = 11
        pos_candidate = np.zeros((n_candiate, 2), dtype=np.int32)
        ind_pos_candidate = np.zeros(n_candiate, dtype=np.int32)
        turbine_pos = power_order[i, 0]
        ind_can = 0
        while True:
            null_turbine_pos = np.random.randint(0, cols * rows)
            if pop[i, null_turbine_pos] == 0:
                pos_candidate[ind_can, 1] = int(np.floor(null_turbine_pos / cols))
                pos_candidate[ind_can, 0] = int(np.floor(null_turbine_pos - pos_candidate[ind_can, 1] * cols))
                ind_pos_candidate[ind_can] = null_turbine_pos
                ind_can += 1
                if ind_can == n_candiate:
                    break
        mars_val = mars.predict(pos_candidate)
        mars_val = mars_val[:, 0]
        sorted_index = np.argsort(-mars_val)  # fitness value descending from largest to least
        null_turbine_pos = ind_pos_candidate[sorted_index[5]]
        pop[i, turbine_pos] = 0
        pop[i, null_turbine_pos] = 1
        power_order[i, 0] = null_turbine_pos
        pop_indices[i, :] = np.sort(power_order[i, :])
        return
    # SUGGA mutation
    def sugga_mutation(self, rows, cols, N, pop,pop_NA, pop_indices, pop_size, mutation_rate):
        np.random.seed(seed=int(time.time()))
        for i in range(pop_size):
            if np.random.randn() > mutation_rate:
                continue
            while True:
                turbine_pos = np.random.randint(0, cols * rows)
                if pop_NA[i, turbine_pos] == 1:
                    break
            while True:
                null_turbine_pos = np.random.randint(0, cols * rows)
                if pop_NA[i, null_turbine_pos] == 0:
                    break
            pop[i, turbine_pos] = 0
            pop[i, null_turbine_pos] = 1

            pop_NA[i, turbine_pos] = 0
            pop_NA[i, null_turbine_pos] = 1

            for j in range(N):
                if pop_indices[i, j] == turbine_pos:
                    pop_indices[i, j] = null_turbine_pos
                    break
            pop_indices[i, :] = np.sort(pop_indices[i, :])
        return

    # SUGGA crossover
    def sugga_crossover(self, N, pop,pop_NA, pop_indices, pop_size, n_parents,
                                     parent_layouts,parent_layouts_NA, parent_pop_indices):
        n_counter = 0
        np.random.seed(seed=int(time.time()))  # init random seed
        while n_counter < pop_size:
            male = np.random.randint(0, n_parents)
            female = np.random.randint(0, n_parents)
            if male != female:
                cross_point = np.random.randint(1, N)
                if parent_pop_indices[male, cross_point - 1] < parent_pop_indices[female, cross_point]:
                    pop[n_counter, :] = 0
                    pop[n_counter, :parent_pop_indices[male, cross_point - 1] + 1] = parent_layouts[male,
                                                                                     :parent_pop_indices[
                                                                                          male, cross_point - 1] + 1]
                    pop[n_counter, parent_pop_indices[female, cross_point]:] = parent_layouts[female,
                                                                               parent_pop_indices[female, cross_point]:]

                    pop_NA[n_counter, :] = pop[n_counter, :]
                    for i in self.NA_loc:
                        pop_NA[n_counter, i - 1] = 2

                    pop_indices[n_counter, :cross_point] = parent_pop_indices[male, :cross_point]
                    pop_indices[n_counter, cross_point:] = parent_pop_indices[female, cross_point:]
                    n_counter += 1
        return
    # SUGGA select
    def sugga_select(self, pop,pop_NA, pop_indices, pop_size, elite_rate, random_rate):
        n_elite = int(pop_size * elite_rate)
        parents_ind = [i for i in range(n_elite)]
        np.random.seed(seed=int(time.time()))  # init random seed
        for i in range(n_elite, pop_size):
            if np.random.randn() < random_rate:
                parents_ind.append(i)
        parent_layouts = pop[parents_ind, :]
        parent_layouts_NA = pop_NA[parents_ind, :]
        parent_pop_indices = pop_indices[parents_ind, :]
        return len(parent_pop_indices), parent_layouts, parent_layouts_NA, parent_pop_indices
    #calculate fitness value
    def sugga_fitness(self, pop, rows, cols, pop_size, N, po):
        fitness_val = np.zeros(pop_size, dtype=np.float32)
        for i in range(pop_size):

            # layout = np.reshape(pop[i, :], newshape=(rows, cols))
            xy_position = np.zeros((2, N), dtype=np.float32)  # x y position
            cr_position = np.zeros((2, N), dtype=np.int32)  # column row position
            ind_position = np.zeros(N, dtype=np.int32)
            ind_pos = 0
            for ind in range(rows * cols):
                if pop[i, ind] == 1:
                    r_i = np.floor(ind / cols)
                    c_i = np.floor(ind - r_i * cols)
                    cr_position[0, ind_pos] = c_i
                    cr_position[1, ind_pos] = r_i
                    xy_position[0, ind_pos] = c_i * self.cell_width + self.cell_width_half
                    xy_position[1, ind_pos] = r_i * self.cell_width + self.cell_width_half
                    ind_position[ind_pos] = ind
                    ind_pos += 1
            lp_power_accum = np.zeros(N, dtype=np.float32)  # a specific layout power accumulate
            for ind_t in range(len(self.theta)):
                for ind_v in range(len(self.velocity)):
                    # print(theta[ind_t])
                    # print(np.cos(theta[ind_t]))
                    trans_matrix = np.array(
                        [[np.cos(self.theta[ind_t]), -np.sin(self.theta[ind_t])],
                         [np.sin(self.theta[ind_t]), np.cos(self.theta[ind_t])]],
                        np.float32)

                    trans_xy_position = np.matmul(trans_matrix, xy_position)
                    speed_deficiency = self.wake_calculate(trans_xy_position, N)

                    actual_velocity = (1 - speed_deficiency) * self.velocity[ind_v]
                    lp_power = self.layout_power(actual_velocity,
                                                     N)  # total power of a specific layout specific wind speed specific theta
                    lp_power = lp_power * self.f_theta_v[ind_t, ind_v]
                    lp_power_accum += lp_power

            sorted_index = np.argsort(lp_power_accum)  # power from least to largest
            po[i, :] = ind_position[sorted_index]

            fitness_val[i] = np.sum(lp_power_accum)
            #
        return fitness_val




class GE_1_5_sleTurbine:
    hub_height = 100.0  # unit (m)
    rator_diameter = 100.0  # unit m
    surface_roughness = 0.25 * 0.001  # unit mm surface roughness
    # surface_roughness = 0.25  # unit mm surface roughness
    rator_radius = 0

    entrainment_const = 0

    def __init__(self):
        self.rator_radius = self.rator_diameter / 2
        self.entrainment_const = 0.5 / np.log(self.hub_height / self.surface_roughness)

        f = open("../../data/power_curve.csv").readlines()
        self.power_curve = {}
        for line in f[1:]:
            line = line.strip().split(',')
            self.power_curve[float(line[0])] = float(line[2])
        return

    # power curve
    # def P_i_X(self, v):
    #     if v < 2.0:
    #         return 0
    #     elif v < 12.8:
    #         return 0.3 * v ** 3
    #     elif v < 18:
    #         return 629.1
    #     else:
    #         return 0
    def P_i_X(self, v):
        if(v<=0):
            return 0
        elif(v>=50):
            return 0
        else:
            try:
                return self.power_curve[math.floor(v*10)/10]
            except:
                import pdb
                pdb.set_trace()





class LayoutGridMCGenerator:
    def __init__(self):
        return

    # rows : number of rows in wind farm
    # cols : number of columns in wind farm
    # n : number of layouts
    # N : number of turbines
    def gen_mc_grid(rows, cols, n, N, lofname):  # , xfname): generate monte carlo wind farm layout grids
        np.random.seed(seed=int(time.time()))  # init random seed
        layouts = np.zeros((n, rows * cols), dtype=np.int32)  # one row is a layout
        # layouts_cr = np.zeros((n*, 2), dtype=np.float32)  # layouts column row index
        positionX = np.random.randint(0, cols, size=(N * n * 2))
        positionY = np.random.randint(0, rows, size=(N * n * 2))
        ind_rows = 0  # index of layouts from 0 to n-1
        ind_pos = 0  # index of positionX, positionY from 0 to N*n*2-1
        # ind_crs = 0
        while ind_rows < n:
            layouts[ind_rows, positionX[ind_pos] + positionY[ind_pos] * cols] = 1
            if np.sum(layouts[ind_rows, :]) == N:
                # for ind in range(rows * cols):
                #     if layouts[ind_rows, ind] == 1:
                #         r_i = np.floor(ind / cols)
                #         c_i = np.floor(ind - r_i * cols)
                #         layouts_cr[ind_crs, 0] = c_i
                #         layouts_cr[ind_crs, 1] = r_i
                #         ind_crs += 1
                ind_rows += 1
            ind_pos += 1
            if ind_pos >= N * n * 2:
                print("Not enough positions")
                break
        # filename = "positions{}by{}by{}N{}.dat".format(rows, cols, n, N)
        np.savetxt(lofname, layouts, fmt='%d', delimiter="  ")
        # np.savetxt(xfname, layouts_cr, fmt='%d', delimiter="  ")
        return layouts

    # rows : number of rows in wind farm
    # cols : number of columns in wind farm
    # n : number of layouts
    # N : number of turbines
    # NA_loc : not usable locations
    # generate layouts with not usable locations
    def gen_mc_grid_with_NA_loc(rows, cols, n, N,NA_loc, lofname,loNAfname):  # , xfname): generate monte carlo wind farm layout grids
        np.random.seed(seed=int(time.time()))  # init random seed
        layouts = np.zeros((n, rows * cols), dtype=np.int32)  # one row is a layout, NA loc is 0

        layouts_NA= np.zeros((n, rows * cols), dtype=np.int32)  # one row is a layout, NA loc is 2
        for i in NA_loc:
            layouts_NA[:,i-1]=2

        # layouts_cr = np.zeros((n*, 2), dtype=np.float32)  # layouts column row index
        positionX = np.random.randint(0, cols, size=(N * n * 2))
        positionY = np.random.randint(0, rows, size=(N * n * 2))
        ind_rows = 0  # index of layouts from 0 to n-1
        ind_pos = 0  # index of positionX, positionY from 0 to N*n*2-1
        # ind_crs = 0
        N_count=0
        while ind_rows < n:
            cur_state=layouts_NA[ind_rows, positionX[ind_pos] + positionY[ind_pos] * cols]
            if cur_state!=1 and cur_state!=2:
                layouts[ind_rows, positionX[ind_pos] + positionY[ind_pos] * cols]=1
                layouts_NA[ind_rows, positionX[ind_pos] + positionY[ind_pos] * cols] = 1
                N_count+=1
                if np.sum(layouts[ind_rows, :]) == N:
                    ind_rows += 1
                    N_count=0
            ind_pos += 1
            if ind_pos >= N * n * 2:
                print("Not enough positions")
                break
        # filename = "positions{}by{}by{}N{}.dat".format(rows, cols, n, N)
        np.savetxt(lofname, layouts, fmt='%d', delimiter="  ")
        np.savetxt(loNAfname, layouts_NA, fmt='%d', delimiter="  ")
        # np.savetxt(xfname, layouts_cr, fmt='%d', delimiter="  ")
        return layouts,layouts_NA

    # generate population
    def gen_pop(rows, cols, n,
                N):  # generate population very similar to gen_mc_grid, just without saving layouts to a file
        np.random.seed(seed=int(time.time()))
        layouts = np.zeros((n, rows * cols), dtype=np.int32)
        positionX = np.random.randint(0, cols, size=(N * n * 2))
        positionY = np.random.randint(0, rows, size=(N * n * 2))
        ind_rows = 0
        ind_pos = 0

        while ind_rows < n:
            layouts[ind_rows, positionX[ind_pos] + positionY[ind_pos] * cols] = 1
            if np.sum(layouts[ind_rows, :]) == N:
                ind_rows += 1
            ind_pos += 1
            if ind_pos >= N * n * 2:
                print("Not enough positions")
                break
        return layouts

    # rows : number of rows in wind farm
    # cols : number of columns in wind farm
    # n : number of layouts
    # N : number of turbines
    # NA_loc : not usable locations
    # generate layouts with not usable locations
    def gen_pop_with_NA_loc(rows, cols, n, N, NA_loc):
        np.random.seed(seed=int(time.time()))  # init random seed
        layouts = np.zeros((n, rows * cols), dtype=np.int32)  # one row is a layout, NA loc is 0

        layouts_NA = np.zeros((n, rows * cols), dtype=np.int32)  # one row is a layout, NA loc is 2
        for i in NA_loc:
            layouts_NA[:, i - 1] = 2

        # layouts_cr = np.zeros((n*, 2), dtype=np.float32)  # layouts column row index
        positionX = np.random.randint(0, cols, size=(N * n * 2))
        positionY = np.random.randint(0, rows, size=(N * n * 2))
        ind_rows = 0  # index of layouts from 0 to n-1
        ind_pos = 0  # index of positionX, positionY from 0 to N*n*2-1
        # ind_crs = 0
        N_count = 0
        while ind_rows < n:
            cur_state = layouts_NA[ind_rows, positionX[ind_pos] + positionY[ind_pos] * cols]
            if cur_state != 1 and cur_state != 2:
                layouts[ind_rows, positionX[ind_pos] + positionY[ind_pos] * cols] = 1
                layouts_NA[ind_rows, positionX[ind_pos] + positionY[ind_pos] * cols] = 1
                N_count += 1
                if np.sum(layouts[ind_rows, :]) == N:
                    ind_rows += 1
                    N_count = 0
            ind_pos += 1
            if ind_pos >= N * n * 2:
                print("Not enough positions")
                break
        return layouts, layouts_NA


