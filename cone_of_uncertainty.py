#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 17:28:29 2019

@author: max
"""
from mapsynth2 import make_synthetic_dataset
import spread_functions as sp
from mapsynth2 import burn_landscapes
from mapsynth2 import burn_landscape_n_times
from evolve2 import multi_sim_main
from evolve2 import main as sim_main
from wildfire import simulate_test_fire
import matplotlib.pyplot as plt
import numpy as np
import sys
from mapsynth2 import make_landscapes 
from wildfire import iou_fitness
from wildfire import make_landscape_from_dir
from matplotlib.colors import ListedColormap
from mapsynth2 import make_landscapes 
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def spread_uphill_only(fire, tree, z, temp, hum, mean_temp, mean_hum, mean_weighted_fire, wind_term):
    if mean_weighted_fire > 0:
        prob = 1
    else:
        prob = 0.05
    return(prob)
    
def generic_spreading(fire, tree, z, temp, hum, mean_temp, mean_hum, mean_weighted_fire, wind_term):
    
    
    

def main():
    # n = number of simulations
    # L = dimension of grid
    # f = mean of noise dist
    # scale = number of perlin iterations
    # max_time = time steps to run fire
    
    # generate 5 simulation landscapes 
    num_gens = 20
    parent_dir_train = 'data/synth_data/simulations/train_data_set'
    n = 3
    L = 100
    f = 10
    scale = 500
    max_time = 40
    
    make_synthetic_dataset(sp.spread_uphill_only,parent_dir_train, n, L, f, scale, max_time, seed=None)
    
    
    
    #--------------------------------------------------------------- TRAIN GP ---------------------------------------------------------------
    # fetch file paths (must change this manually to run a different expirement)
    path_head = 'data/synth_data/simulations/train_data_set/plain_n_'+str(n)+'_L_'+str(L)+'_f_'+str(f)+'_scale_'+str(scale)+'_max_time_'+str(max_time)+'_seed_None/landscape_'
    
    init_landscape_paths = []
    final_landscape_paths = []
    for i in range(n):
        init_landscape_paths.append(path_head+str(i)+'/init_landscape.npy')
        final_landscape_paths.append(path_head+str(i)+'/final_landscape.npy')
    
    # run evolution
    best_ind,log,fitnessess = multi_sim_main(init_landscape_paths, final_landscape_paths, max_time)
    
    
    # --------------------------------------------------------------- REPORT TRAINING STATS ---------------------------------------------------------------
    # generate hist of fitnessess
    plt.hist(fitnessess)
    
    #caclulate avg fitness per generation
    
    numevals = []
    for i in log:
        numevals.append(i['nevals'])
        
    avg_fitness = []
    start_idx = 0 
    
    cumulative_func_evals = []

    for idx,n in enumerate(numevals):
        print(start_idx , n +start_idx)
        avg_fitness.append(np.average(fitnessess[start_idx:n+start_idx]))
        start_idx = start_idx + n
        cumulative_func_evals.append(start_idx)
    
    plt.figure(figsize = (10,10))
    plt.plot(numevals,avg_fitness)
    plt.title("Number of Evals vs. Avg Generational Fitness")
    plt.xlabel("Number of Evaluations")
    plt.ylabel("Avg. Fitness")
    plt.savefig("evalsVsFitness.jpg")
    
    plt.figure(figsize = (10,10))
    plt.plot(cumulative_func_evals,avg_fitness)
    plt.title("Cummulative Number of Functiona Evals. Avg Generational Fitness")
    plt.xlabel("Number of Evaluations")
    plt.ylabel("Avg. Fitness")
    plt.savefig("cumEvalsVsFitness.jpg")
    

    #--------------------------------------------------------------- TEST GP ---------------------------------------------------------------
    
    # take best simulation, run it on the same fire 10 times to build the cone of uncertainty
    
    
    landscape_dir = 'data/synth_data/simulations/train_data_set/plain_n_'+str(n)+'_L_'+str(L)+'_f_'+str(f)+'_scale_'+str(scale)+'_max_time_'+str(max_time)+'_seed_None/landscape_2'
    
    num_burns = 10
    burn_landscape_n_times(spread_uphill_only,dataset_dir=None, landscape_dir=landscape_dir, max_time=65, num_periods=num_burns, seed=None, display=True)
    
    #build plot from simulation runs
    final_landscapes = []
    for burn in range(num_burns):
        burn_landscape = np.load(landscape_dir + "/landscape_burn_0{}.npy".format(burn))
        final_landscapes.append(burn_landscape)
        


    
    
    fig = plt.figure(figsize = [10,10])
    ax = fig.add_subplot(111)
    
    cmap = ListedColormap(['g', 'r'])
    for final_landscape in final_landscapes:
        cax = ax.matshow(final_landscape[:,:,0],cmap=cmap, alpha = .15)
        contours = plt.contour(final_landscape[:,:,2])
        plt.title("Cone of Uncertainty over {} Simulations".format(num_burns))
        
        
#        radians = final_landscape[:,:,6]
#        x = np.arange(0,100,1)[::10]
#        y = np.arange(0,100,1)[::10]
#        X, Y = np.meshgrid(x, y)
#        u = np.cos(radians)[::10,::10]
#        v = np.sin(radians)[::10,::10]
#        fig, ax = plt.subplots(figsize=(20,20))
#        ax.quiver(X,Y,u,v)
        plt.clabel(contours, inline=1, fontsize=10)

        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.set_aspect('equal')
        
    

    plt.show()
    plt.savefig("cone_of_uncertainty.jpg")

     

main()


def cone_of_uncertainty(landscape_dir,num_burns,spreading_func):
    
    burn_landscape_n_times(spreading_func,dataset_dir=None, landscape_dir=landscape_dir, max_time=65, num_periods=num_burns, seed=None, display=False)
    
    #build plot from simulation runs
    final_landscapes = []
    for burn in range(num_burns):
        burn_landscape = np.load(landscape_dir + "/landscape_burn_0{}.npy".format(burn))
        final_landscapes.append(burn_landscape)
        

    fig = plt.figure(figsize = [10,10])
    ax = fig.add_subplot(111)
    
    cmap = ListedColormap(['g', 'r'])
    for final_landscape in final_landscapes:
        cax = ax.matshow(final_landscape[:,:,0],cmap=cmap, alpha = .15)
        contours = plt.contour(final_landscape[:,:,2])
        plt.title("Cone of Uncertainty over {} Simulations".format(num_burns))
        
        
#        radians = final_landscape[:,:,6]
#        x = np.arange(0,100,1)[::10]
#        y = np.arange(0,100,1)[::10]
#        X, Y = np.meshgrid(x, y)
#        u = np.cos(radians)[::10,::10]
#        v = np.sin(radians)[::10,::10]
#        fig, ax = plt.subplots(figsize=(20,20))
#        ax.quiver(X,Y,u,v)
        plt.clabel(contours, inline=1, fontsize=10)

        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.set_aspect('equal')
        
    plt.show()
    fig.savefig("cone_of_uncertainty.jpg")

landscape_dir = 'data/synth_data/simulations/train_data_set/plain_n_'+str(n)+'_L_'+str(L)+'_f_'+str(f)+'_scale_'+str(scale)+'_max_time_'+str(max_time)+'_seed_None/landscape_2'

cone_of_uncertainty(landscape_dir,10,spread_uphill_only)



