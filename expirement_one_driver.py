#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 17:28:29 2019
@author: max
"""
from mapsynth import make_synthetic_dataset,burn_landscape_n_times, burn_landscapes,make_landscapes 
import spread_functions as sp

from evolve import multi_sim_main,multi_sim_main_rand

import matplotlib.pyplot as plt
import numpy as np
from wildfire import iou_fitness, multi_sim_iou_fitness
from matplotlib.colors import ListedColormap
import math
import features

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def make_fitness_boxplot(fitnessess):
    fig = plt.figure(1, figsize=(9, 6))

    # Create an axes instance
    ax = fig.add_subplot(111)
    
    # Create the boxplot
    
    
    bp = ax.boxplot(fitnessess, patch_artist=True)
    
    ## change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set( color='#7570b3', linewidth=2)
        # change fill color
        box.set( facecolor = '#1b9e77' )
    
    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)
    
    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)
    
    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)
    
    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)
        # Save the figure
        
    ax.set_xticklabels(['Random Expression', 'Best Expression','Wind Logit','Elevation Logit','Balanced Logit'])
    plt.title("Fitness Distributions")
    fig.savefig('boxplot_fitnessess.png', bbox_inches='tight')
    
    
def cone_of_uncertainty(landscape_dir,num_burns,spreading_func,func_type):

    burn_landscape_n_times(spreading_func,dataset_dir=None, landscape_dir=landscape_dir, max_time=65, num_periods=num_burns, seed=None, display=False,func_type = func_type)
    
    #build plot from simulation runs
    final_landscapes = []
    for burn in range(num_burns):
        burn_landscape = np.load(landscape_dir + "/landscape_burn_{}_{}.npy".format(func_type,burn))
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


def expirement_one_driver(n,L,f,scale,num_gens,max_time,parent_dir,base_spread_func,base_func_type):
    
    
    # n = number of simulations
    # L = dimension of grid
    # f = mean of noise dist
    # scale = number of perlin iterations
    # max_time = time steps to run fire
    
    # generate 5 simulation landscapes 
   
    # make the synthetic dataset for the simulations
    
    make_synthetic_dataset(base_spread_func,parent_dir, n, L, f, scale, max_time, func_type=base_func_type, seed=None)
    

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
    
    # generate random tree
    rand_ind,rand_log,rand_fitnessess = multi_sim_main_rand(init_landscape_paths, final_landscape_paths, max_time)

    
    # --------------------------------------------------------------- REPORT TRAINING STATS ---------------------------------------------------------------
    
    
    numevals = []
    for i in log:
        numevals.append(i['nevals'])
        
    avg_fitness = []
    start_idx = 0 
    
    cumulative_func_evals = []

    for idx,n in enumerate(numevals):
        avg_fitness.append(np.average(fitnessess[start_idx:n+start_idx]))
        start_idx = start_idx + n
        cumulative_func_evals.append(start_idx)
    
    plt.figure(figsize = (10,10))
    plt.plot(numevals,avg_fitness)
    plt.title("Number of Evals vs. Avg Generational Fitness")
    plt.xlabel("Number of Evaluations")
    plt.ylabel("Avg. Fitness")
    
    
    plt.figure(figsize = (10,10))
    plt.plot(cumulative_func_evals,avg_fitness)
    plt.title("Cummulative Number of Functiona Evals. Avg Generational Fitness")
    plt.xlabel("Number of Evaluations")
    plt.ylabel("Avg. Fitness")
    

    #--------------------------------------------------------------- TEST GP ---------------------------------------------------------------
    
    # run test against another set of simulations
    
    parent_dir_test = 'data/synth_data/simulations/test_data_set/plain_n_'+str(n)+'_L_'+str(L)+'_f_'+str(f)+'_scale_'+str(scale)+'_max_time_'+str(max_time)+'_seed_None'


    # generate n landscapes to spread fire on 
    make_landscapes(parent_dir_test,n,L,scale,seed=None,display=False)
    


    # run n simulations on new landscapes using desired spreading functions
    
    
    burn_landscapes(best_ind,dataset_dir=parent_dir_test, landscape_dir=None, max_time=60, num_periods=1, seed=None, display=True, func_type = 'best')
    
    burn_landscapes(rand_ind,dataset_dir=parent_dir_test, landscape_dir=None, max_time=60, num_periods=1, seed=None, display=True, func_type='rand')

    burn_landscapes(sp.spread_uphill_only,dataset_dir=parent_dir_test, landscape_dir=None, max_time=60, num_periods=1, seed=None, display=True, func_type='balanced_logits')

    burn_landscapes(sp.spread_uphill_only,dataset_dir=parent_dir_test, landscape_dir=None, max_time=60, num_periods=1, seed=None, display=True, func_type='dz_logits')

    burn_landscapes(sp.spread_uphill_only,dataset_dir=parent_dir_test, landscape_dir=None, max_time=60, num_periods=1, seed=None, display=True, func_type='wind_logits')

    

    rand_fitnessess = []
    best_fitnessess = []
    wind_fitnessess = []
    dz_fitnessess = []
    balanced_fitnessess = []

    for sim_num in range(n):
        pred_landscape_best= np.load(parent_dir_test+"/landscape_"+str(sim_num)+"/landscape_burn_best_1.npy")
        pred_landscape_rand = np.load(parent_dir_test+"/landscape_"+str(sim_num)+"/landscape_burn_rand_1.npy")
        pred_landscape_wind = np.load(parent_dir_test+"/landscape_"+str(sim_num)+"/landscape_burn_wind_logits_1.npy")
        pred_landscape_balanced = np.load(parent_dir_test+"/landscape_"+str(sim_num)+"/landscape_burn_balanced_logits_1.npy")
        pred_landscape_dz = np.load(parent_dir_test+"/landscape_"+str(sim_num)+"/landscape_burn_dz_logits_1.npy")

        # if ground truth was generated form balanced logit then : 
        true_landscape = pred_landscape_balanced

        rand_fitnessess.append(iou_fitness(true_landscape, pred_landscape_rand))
        best_fitnessess.append(iou_fitness(true_landscape, pred_landscape_best))
        wind_fitnessess.append(iou_fitness(true_landscape, pred_landscape_wind))
        dz_fitnessess.append(iou_fitness(true_landscape, pred_landscape_dz))
        balanced_fitnessess.append(iou_fitness(true_landscape, pred_landscape_balanced))



        
    
    # calculate test fitness of simulation
    #rand_avg_fitness = np.average(rand_fitnessess)
    #best_avg_fitness = np.average(best_fitnessess)
    
    #print("avg fitness of random tree",rand_avg_fitness)
    #print("avg fitness of best tree",best_avg_fitness)
    
    
    # make boxplot of fitnessess
    fitnessess = [rand_fitnessess,best_fitnessess,wind_fitnessess,dz_fitnessess,balanced_fitnessess]
    make_fitness_boxplot(fitnessess)
    
    # produce cone of uncertainty to any of the functions on a given landscape 
    prob_func = features.make_logits_fire_func(features.make_scaled_logits_func(bias=-7, feature_scale=np.array([6.0, 0.2, 0.2, 0.8])))
    cone_of_uncertainty(parent_dir_test+'/landscape_0',6,prob_func,'balanced_logits')
    

def main():
# define path and params
    parent_dir = 'data/synth_data/simulations/train_data_set'
    n = 3
    L = 100
    f = 10
    scale = 500
    max_time = 40
    num_gens = 20
    base_spread_func = sp.spread_uphill_only
    base_func_type='balanced_logit'
    
    expirement_one_driver(n,L,f,scale,num_gens,max_time,parent_dir,base_spread_func,base_func_type)
     
main()
