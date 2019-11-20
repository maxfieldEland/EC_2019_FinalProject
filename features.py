
import numpy as np
from scipy.special import expit
from wildfire import L_FIRE, L_TREE, L_Z, L_TEMP, L_HUM, L_WS, L_WD, N_CENTER
import math

def make_scaled_logits_func(bias, feature_scale):
    '''
    Return a function that takes the "logits" features from `make_logits_fire_func` and returns
    the probability of a cell transitioning to the fire state. Specifically the function returns:

    logits = bias + feature_scale.dot(features)
    probability = logistic(logits)

    :param bias: Used to shift the probabilities higher and lower overall.
    :param feature_scale: Has a weight for each feature from `make_logits_fire_func`: dz_fire_logits,
    temp_logits, hum_logits, wind_fire_logits. Feature scale changes the sensitivity of fire to different
    neighborhood features.
    :return: probability that the cell transitions to the fire state.
    '''
    def func(features):
        '''
        Compute the probability by taking a weighted sum of the parameters using the parameter scales.
        Return the result of the logistic function called with the weighted sum, the logits.

        :param ndarray features: The features from `make_logits_fire_func`,
         dz_fire_logits, temp_logits, hum_logits, wind_fire_logits
        :return: The probability of a cell transitioning to fire.
        '''
        # print('features:', features)
        logits = feature_scale * features
        # print('logits:', logits)
        prob = expit(bias + logits.sum())  # logistic function of weighted sum
        return prob

    return func


def make_logits_fire_func(func):
    '''
    Return a function that takes a neighborhood, transforms it into our hand-engineered
    features, passes those features (as an ndarray) to func and returns what func returns.

    :param func: A function that takes the engineered features and returns the probability of fire spreading.
    '''

    # The effect of the northward component of wind. The magnitude represent the northward speed of the wind.
    # The sign indicates weather northward is toward center (1), away from center (-1) or parallel to center (0).
    # Since each cell is a neighborhood is oriented toward the center in a different manner, the cells have
    # different weights:
    # center cell = 0.0 b/c the center cell is not on fire so this won't matter.
    # north cell = -1.0 b/c in the north cell, a northward wind blows fire away from the center
    # east cell = 0.0 b/c in the east cell a northward wind does not blow fire away or toward the center
    # south cell = 1.0 b/c in the south cell, a northward wind blows fire toward the center
    # west cell = 0.0 b/c in the west cell, a northward wind does not blow fire away or toward the center
    northward_effect = np.array([0.0, -1.0, 0.0, 1.0, 0.0])
    # The effect of the northward component of wind.
    eastward_effect = np.array([0.0, 0.0, -1.0, 0.0, 1.0])

    def prob_func(neighborhood):
        '''
        Compute linear logits for temperature and humidity, ranging from 0 to 11 mostly uniformly.
        Compute logits from dz as the sum of the logits of the on-fire neighbors, where dz is scaled such that
        :param neighborhood: shape = (n_neighbors, n_layers) = (5, 7). The rows are in the order: center,
         north, east, south, west. Cells on the edge of a landscape have nan values for neighbors that
         are out-of-bounds.
        :return: the probability of fire spreading to cell.
        '''
        idx = ~np.isnan(neighborhood[:, L_FIRE])  # boolean idx of rows that are in bounds

        # z is composed of a landscape wide base value in [0, 10) + perlin noise in [0, 1]
        # temperature is composed of a landscape wide base value in [0, 10) + perlin noise in [0, 1]
        # humidity is composed of a landscape wide base value in [0, 10) + perlin noise in [0, 1]
        # wind speed is composed of a landscape wide base value in [0, 10) + perlin noise in [0, 1]
        # wind direction is composed of a landscape wide base value in [0, 2pi) + perlin noise in [-pi/8, pi/8]

        # fire spreads uphill faster than downhill
        # the vertical distance from cell to neighbor
        dz = neighborhood[:, L_Z] - neighborhood[N_CENTER, L_Z]
        # Regarding the range of delta_z, because delta_z subtracts two cell values, the base value is removed.
        # Only the Perlin noise remains. So delta_z could theoretically range from -1 to 1.
        # But because of smoothing, the actual range is between [-0.03, 0.03] and mostly almost 0.
        # let's say we want a -delta_z of 0.01 to correspond to 2 logits instead of 1 logit. Then
        # np.exp(0.01 * 69.3147) == 2.0, np.exp(0.02 * 69.3147) == 4.0, and np.exp(0.03 * 69.3147) = 8.0 logits.
        dz_factor = 69.3147
        dz_logits = np.exp(-dz * dz_factor)
        # every on fire cell additively contributes to the logits scaled by the difference in elevation.
        dz_fire_logits = np.nansum(neighborhood[:, L_FIRE] * dz_logits)
        # I think dz_fire_logits will range from ~1/8 when one steeply uphill neighbor is on fire to ~32 when four
        # steeply downhill neighbors are on fire. The median will be ~1 to 4, about 1 logit per on-fire neighbor.
        # fire spreads with the wind faster than against the wind
        # distribution: between ~1/8 to ~32, typically around 1

        # higher temperature => higher chance of catching on fire
        temp_logits = np.nanmean(neighborhood[:, L_TEMP])
        # distribution: between 0 and 11 logits, ~5.5 avg

        # higher humidity => lower chance of catching on fire
        hum_logits = -np.nanmean(neighborhood[:, L_HUM])
        # distribution: between 0 and -11 logits ~ -5.5 avg

        eastward_wind = neighborhood[:, L_WS] * np.cos(neighborhood[:, L_WD])  # positive if wind blowing east
        northward_wind = neighborhood[:, L_WS] * np.sin(neighborhood[:, L_WD])  # positive if wind blowing north
        # wind_toward_center is positive if wind blowing toward the neighborhood center, negative if blowing away
        wind_toward_center = northward_wind * northward_effect + eastward_wind * eastward_effect

        # Wind can range from -11 away from fire to 11 toward fire.
        # Wind blowing fire toward the center increases the probability of fire spreading to the center
        # Wind blowing fire away reduces the probability.
        # Assume every neighbor independently and additively affects the probability of fire and that
        # probability can be expressed as a number in logits.
        wind_factor = 2.0794415416798357 / 11  # wind of 11 scales to 8 logits
        wind_logits = np.exp(wind_toward_center * wind_factor)
        wind_fire_logits = np.nansum(neighborhood[:, L_FIRE] * wind_logits)
        # I think the wind_fire_logits will range from 1/8 when only one cell is on fire and the wind is blowing strongly
        # away from center to around 10 when all neighboring cells are on fire and the wind is blowing strongly.
        # I think the median will be around 1 logit if one neighbor is burning parallel to the wind to around 5
        # if 4 neighbors are burning and the wind is blowing speed 5
        # distribution: around ~1/8 to 10. Typically around 1 logit.

        return func(features=np.array((dz_fire_logits, temp_logits, hum_logits, wind_fire_logits), dtype=float))

    return prob_func

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