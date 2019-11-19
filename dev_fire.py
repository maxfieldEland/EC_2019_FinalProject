#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 10:51:19 2019

@author: max
"""

from mapsynth import make_synthetic_dataset
from evolve import multi_sim_main
from evolve import main as sim_main

import matplotlib.pyplot as plt

def main():
    # n = number of simulations
    # L = dimension of grid
    # f = mean of noise dist
    # scale = number of perlin iterations
    # max_time = time steps to run fire
    
    # generate 5 simulation landscapes 
    num_gens = 20
    parent_dir_train = 'data/synth_data/simulations/train_data_set'
    n = 2
    L = 50
    f = 10
    scale = 100
    max_time = 20
    
    #make_synthetic_dataset(parent_dir_train, n, L, f, scale, max_time, seed=None)
        # fetch file paths (must change this manually to run a different expirement)
    path_head = 'data/synth_data/simulations/train_data_set/plain_n_'+str(n)+'_L_'+str(L)+'_f_'+str(f)+'_scale_'+str(scale)+'_max_time_'+str(max_time)+'_seed_None/landscape_'
    
    init_landscape_paths = []
    final_landscape_paths = []
    for i in range(n):
        init_landscape_paths.append(path_head+str(i)+'/init_landscape.npy')
        final_landscape_paths.append(path_head+str(i)+'/final_landscape.npy')
    
    # run evolution
    best_ind,log,fitnessess = multi_sim_main(init_landscape_paths, final_landscape_paths, max_time)
    
    print(fitnessess)
    
main()
