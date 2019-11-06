'''
Create synthetic landscapes, including topology, wind, temperature, and precipitation.
'''

import argparse
import copy
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import wildfire as wf


def fade(t):
    """
    Generate perlin noise_type

    Parameters:
        :int t: value

    Returns:
        output of fade function.
    """

    return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3


def perlin_raster(n_rows, n_cols, f, noise):
    """
    Generate random 2D maps that mimic spatially smoothed features.
    :param int n_rows: the number of rows in the map
    :param int n_cols: the number of columns in the map
    :param int f: a scale factor and the height displacement (phase
    displacement shift/ translation).
    :param str noise: the noise function that defines the deformation
    of a point across the 2D space. 'Sinosoid': generates a sinosoidal
    surface centered at f+1. (see:
    https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html)
    'Saddle'
        :str Perlin: ?


    Returnss:
        2D ndarray describing height of map
    """
    # octaves are the number of waves to combine
    # frequency is the number of oscillations per time step
    # amplitude is the height of any given point on the waveform
    # get row,col:

    if noise == 'Saddle':
        # initialize a meshgrid:
        X, Y = np.mgrid[-1:1:n_rows + 0j, -1:1:n_cols + 0j]
        # calculate heights at each site (centered about Z = f + 1):
        Z = ((f * (X + Y) * np.exp(-f * (X ** 2 + Y ** 2)) + 1) * f + f) + 1
    elif noise == 'Perlin':
        Z = np.zeros((n_rows, n_cols))
        for j in range(n_cols):
            for i in range(n_rows):
                Z[i, j] = fade(np.random.randint(1, f)) / (n_rows * n_cols)

    return Z


def display_raster(raster):
    """
    Graphically display the raster, e.g. the topography of the map.
    """
    n_rows, n_cols = raster.shape
    X, Y = np.mgrid[:n_rows, :n_cols]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    surf = ax.plot_surface(X, Y, raster, cmap='Greens')
    plt.show()


def make_initial_and_final_state(landscape_dir='data/synthetic_example', max_time=20, seed=None):
    '''
    Create a landscape, start an initial fire, simulate a burn for max_time time steps,
    and then save the initial fire and final fire landscapes.
    :param landscape_path: directory containing landscape layer files
    :param seed: optional integer seed
    '''
    if seed is not None:
        np.random.seed(int(seed))

    landscape_path = Path(landscape_dir)
    landscape = wf.make_landscape_from_dir(landscape_path)
    final_landscape = copy.deepcopy(landscape)

    # START FIRE IN CENTER OF LANDSACPE
    i = landscape.shape[0] // 2
    j = landscape.shape[1] // 2
    landscape[i, j, wf.L_STATE] = wf.S_FIRE
    init_landscape = copy.deepcopy(landscape)

    # Simulate burn
    max_time = 20
    gamma = 0.7
    state_maps = wf.simulate_fire(landscape, max_time, wf.make_calc_prob_fire(gamma))
    final_landscape[:, :, wf.L_STATE] = state_maps[-1]

    # save initial and final landscapes
    np.save(landscape_path / 'init_landscape.npy', init_landscape)
    np.save(landscape_path / 'final_landscape.npy', final_landscape)
    wf.show_landscape(init_landscape)
    wf.show_landscape(final_landscape)

    print(wf.loss_function(final_landscape[:, :, wf.L_STATE], final_landscape[:, :, wf.L_STATE]))
    print(wf.loss_function(init_landscape[:, :, wf.L_STATE], final_landscape[:, :, wf.L_STATE]))


def main(*args):
    # raster = perlin_raster(50, 50, f=6, noise='Saddle')
    # raster = perlin_raster(50, 50, f=6, noise='Perlin')
    # raster = np.load('data/perlin_50_x_50_100.npy')
    for fn in ('topology.npy', 'wind_speed.npy', 'temperature.npy', 'humidity.npy'):
        path = Path('data/synthetic_example') / fn
        raster = np.load(path)
        print(f'{fn} shape: {raster.shape}')
        display_raster(raster)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('func', help='a function to run')
    parser.add_argument('args', metavar='ARG', nargs='*', help='any arguments for the function')
    args = parser.parse_args()
    globals()[args.func](*args.args)


