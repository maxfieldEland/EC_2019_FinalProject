#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 09:22:53 2019

FIRE SPREADING FUNCTIONS 
@author: max
"""
import math
import numpy as np
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def spread_uphill_only(fire, tree, z, temp, hum, mean_temp, mean_hum, mean_weighted_fire, wind_term):
    if mean_weighted_fire > 0:
        prob = 1
    else:
        prob = 0.2
    return(prob)
    
    
def generic_spreading(fire, tree, z, temp, hum, mean_temp, mean_hum, mean_weighted_fire, wind_term):
    prod = (wind_term**2+mean_weighted_fire*mean_temp-mean_hum)/(mean_hum**4*(1/wind_term))
 
    return(prod)
    
    
def spread_with_wind(fire, tree, z, temp, hum, mean_temp, mean_hum, mean_weighted_fire, wind_term):
    
    if wind_term > 100:
        wind_prob = .8
    else:
        wind_prob = .2
        
    if mean_weighted_fire > 0:
        z_prob = 0.8
    else:
        z_prob = 0.2
    
    beta = 5
    prob = np.average([beta*wind_prob, z_prob/beta])
    #print(prob)
    return(prob)