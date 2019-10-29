


import wildfire
import spread_fire
import cProfile
from copy import deepcopy
import datetime
import numpy as np


def main():
    n_reps = 20
    max_time = 20
    gamma = 0.7
    z_max = 4
    threshold = 5
    landscape_file = 'data/perlin_50_x_50_100.npy'

    sf_landscape = spread_fire.make_landscape(landscape_file)
    wf_landscape = wildfire.make_landscape(landscape_file)

    # profile spread_fire
    landscape = deepcopy(sf_landscape)
    cProfile.runctx('spread_fire.fire_init(landscape, gamma, z_max, threshold, max_time)', globals(), locals())

    # profile wildfire
    landscape = deepcopy(wf_landscape)
    # Start fire in center of landscape, b/c wildfire.simulate_fire does not.
    i = landscape.shape[0] // 2
    j = landscape.shape[1] // 2
    landscape[i, j, wildfire.L_STATE] = wildfire.S_FIRE
    cProfile.runctx('wildfire.simulate_fire(landscape, gamma, max_time, wildfire.calc_prob_fire)', globals(), locals())

    sf_times = np.empty(n_reps, dtype=float)
    wf_times = np.empty(n_reps, dtype=float)
    for i in range(n_reps):
        print('rep', i)
        landscape = deepcopy(sf_landscape)
        start = datetime.datetime.now()
        spread_fire.fire_init(landscape, gamma, z_max, threshold, max_time)
        sf_times[i] = (datetime.datetime.now() - start) / datetime.timedelta(microseconds=1)

        landscape = deepcopy(wf_landscape)
        # Start fire in center of landscape, b/c wildfire.simulate_fire does not.
        row = landscape.shape[0] // 2
        col = landscape.shape[1] // 2
        landscape[row, col, wildfire.L_STATE] = wildfire.S_FIRE
        start = datetime.datetime.now()
        wildfire.simulate_fire(landscape, gamma, max_time, wildfire.calc_prob_fire)
        wf_times[i] = (datetime.datetime.now() - start) / datetime.timedelta(microseconds=1)

    print('num repetitions:', n_reps)
    print('spread_fire elapsed time:', sf_times.sum())
    print('wildfire elapsed time:', wf_times.sum())
    print('spread_fire mean time:', sf_times.mean())
    print('wildfire mean time:', wf_times.mean())
    print('spread_fire std dev time:', sf_times.std())
    print('wildfire std dev time:', wf_times.std())
    print('spread_fire / wildfire ratio:', sf_times.sum() / wf_times.sum())




if __name__ == '__main__':
    main()

