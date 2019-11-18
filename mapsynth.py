"""
Create synthetic landscapes, including topography, humidity, temperature, wind speed and wind direction.

For each landscape, first a set of layers is generated (one layer for topography, for temperature, etc.).
Then the layers are stitched into a landscape (3-d ndarray), an initial burn is created, a wildfire
simulation is run, and then the initial burned landscape and final burned landscape are saved.

Usage
=====

Make Landscapes
---------------

Make n=2 landscapes, layers only:

python mapsynth.py make_landscapes --dataset-dir=foo -n2 -L100 --scale=400 --seed=1 --display


Burn Landscapes
---------------

Different burn functions are influenced by different factors in the landscape, like wind or difference in elevation (dz)
or a balance of elevation change, temp, humidity, and wind. The original burn function is available too.
Burn every landscape in a dataset for 4 * 20 timesteps, saving the landscape initially and after every 20 time step burn:

time python mapsynth.py burn_landscapes --dataset-dir=foo --max-time=20 --num-periods=4 --seed=1 --display --func-type balanced_logits

time python mapsynth.py burn_landscapes --dataset-dir=foo --max-time=20 --num-periods=4 --seed=1 --display --func-type dz_logits

time python mapsynth.py burn_landscapes --dataset-dir=foo --max-time=20 --num-periods=4 --seed=1 --display --func-type wind_logits

time python mapsynth.py burn_landscapes --dataset-dir=foo --max-time=20 --num-periods=4 --seed=1 --display --func-type original

Visualization
-------------

Display a burned landscape:

python mapsynth.py display_landscape_file --path=foo/landscape_00/landscape_burn_1.npy


"""

import argparse
import copy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import wildfire as wf
from wildfire import L_FIRE, L_TREE, L_Z, L_TEMP, L_HUM, L_WS, L_WD, N_CENTER
import features


def display_layer(layer):
    x, y = np.mgrid[:layer.shape[0], :layer.shape[1]]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(x, y, layer, cmap='Greens')
    plt.show()


def display_layers_from_dir(path):
    '''
    Show all the layers in a landscape directory
    :param path: landscape directory
    '''
    for name in ['topography', 'temperature', 'humidity', 'wind_speed', 'wind_direction']:
        fn = Path(path) / (name + '.npy')
        layer = np.load(fn)
        display_layer(layer)


def display_landscape_file(path):
    landscape = np.load(path)
    wf.show_landscape(landscape)


def plot_landscape_histograms_from_file(path):
    landscape = np.load(path)
    plot_landscape_histograms(landscape)


def plot_landscape_histograms(landscape):
    n_hood = wf.get_neighborhood_values(landscape, 0, 0).shape[0]  # the number of cells in a neighborhood (== 5)
    dzs = np.zeros((landscape.size, n_hood))  # shape: n_landscape, n_hood
    k = 0
    for i in range(landscape.shape[0]):
        for j in range(landscape.shape[1]):
            k += 1
            hood = wf.get_neighborhood_values(landscape, i, j)
            delta_z = hood[:, L_Z] - hood[N_CENTER, L_Z]
            dzs[k, :] = delta_z

    dzs_mean = np.nanmean(dzs)
    dzs_std = np.nanstd(dzs)
    print('dzs_mean:', dzs_mean)
    print('dzs_std:', dzs_std)
    print('thresh, mean, std')
    for thresh in [0.0, 0.0001, 0.001, 0.01, 0.1]:
        m = np.nanmean(dzs.ravel()[np.abs(dzs.ravel()) >= thresh])
        s = np.nanstd(dzs.ravel()[np.abs(dzs.ravel()) >= thresh])
        print(f'{thresh}, {m}, {s}')

    fig, ax = plt.subplots(5, 1, figsize=(12, 9), sharex=True)
    bins = 100
    ax[0].hist(dzs[:, 1:].ravel(), bins=bins)
    ax[0].set_title('all')
    ax[1].hist(dzs[:, 1], bins=bins)
    ax[1].set_title('north')
    ax[2].hist(dzs[:, 2], bins=bins)
    ax[2].set_title('east')
    ax[3].hist(dzs[:, 3], bins=bins)
    ax[3].set_title('south')
    ax[4].hist(dzs[:, 4], bins=bins)
    ax[4].set_title('west')
    plt.suptitle('Delta Z Histogram Segmented by Neighborhood Location')
    plt.show()


def fade(t):
    """
    generate perlin noise_type
    """
    return 6*t**5 - 15*t**4 + 10* t**3


def make_perlin_layer(L, f, scale, norm=True, display=False):
    '''
    Make a 2D perlin noise layer.

    :param L: The size of the layer
    :param f: Approximately the mean value of the layer
    :param scale: The amount of smoothing of the layer
    :param norm: if True, shift and scale the layer values to be between 0 and 1.
    :param display: If True, display the layer. Useful for development.
    :return:
    '''
    # choose different intervals to pick samples from:
    low_high = np.array([(0, 1), (0.25, 0.75), (0.33, 0.67), (0.375, 0.625)])

    # at each site in the grid, initialize some noise
    shape = (L, L)
    x = fade(np.random.random(shape)) * f
    y = fade(np.random.random(shape)) * f
    layer = x + y

    # add padding so each cell in layer has 4 neighbors, even cells on the edges of the landscape.
    # shape (L+2, L+2)
    padded_layer = np.pad(layer, pad_width=((1, 1), (1, 1)), mode='constant', constant_values=np.nan)

    n_idx = get_neighbors_idx(padded_layer)
    n_row_idx = n_idx[0][1:-1, 1:-1, :] # the row index of every neighbor in padded_layer
    n_col_idx = n_idx[1][1:-1, 1:-1, :] # the column index of every neighbor in padded_layer

    for i in range(scale + 1):
        # for each cell, choose a random interval from low_high
        lh = np.random.choice(len(low_high), size=shape)  # matrix of random indices into low_high array
        lows = np.expand_dims(low_high[lh][:, :, 0], -1)  # shape (L, L, 1). expand dims for broadcasting
        highs = np.expand_dims(low_high[lh][:, :, 1], -1)
        # for each neighbor of each cell, uniformly sample from that interval
        rn = np.random.random(shape + (4,)) * (highs - lows) + lows  # rn.shape=(L, L, 4)

        # for each cell, for each neighbor, linearly interpolate between cell and each neighbor by a random amount
        neighbor_values = padded_layer[(n_row_idx, n_col_idx)]  # shape (L, L, 4)
        cell_values = np.expand_dims(padded_layer[1:-1, 1:-1], -1)  # shape (L, L, 1)
        lerps = cell_values + rn * (neighbor_values - cell_values)  # shape (L, L, 4)

        # the new cell values are the mean of the interpolated values
        padded_layer[1:-1, 1:-1] = np.nanmean(lerps, axis=2)

    layer = padded_layer[1:-1, 1:-1]  # remove padding
    if norm:
        layer = layer - np.amin(layer)
        layer = layer / np.amax(layer)

    if display:
        display_layer(layer)

    return layer


def get_neighbors_idx(layer):
    idx = np.indices(layer.shape)
    north = idx.copy()
    north[0] = idx[0] - 1  # north is cell in the row above
    east = idx.copy()
    east[1] = idx[1] + 1  # east is the cell in the column to the right
    south = idx.copy()
    south[0] = idx[0] + 1  # south is the cell in the row below
    west = idx.copy()
    west[1] = idx[1] - 1  # west is the cell in the column to the left
    stacked = np.stack([north, east, south, west], axis=3)  # axes: (row/col, n_row, n_col, n_neighborhood)
    return stacked


def make_layer(path, name, L=50, scale=100, signal_range=(0., 10.), noise_range=(0., 1.), display=False):
    '''
    A layer is a matrix of values composed of a signal value + scaled, shifted perlin noise.

    :param path:
    :param name:
    :param L:
    :param scale: the smoothness of the perlin noise. L and scale interact to determine smoothness.
    As L increases, scale has to increase to maintain similarly
    smooth contours at the larger landscape size.
    :param signal_range:
    :param noise_range:
    :param display:
    :return:
    '''
    # noise
    noise_min, noise_max = noise_range
    noise = make_perlin_layer(L, f=10, scale=scale, norm=True) * (noise_max - noise_min) + noise_min

    signal_min, signal_max = signal_range
    signal = np.random.random() * (signal_max - signal_min) + signal_min
    layer = signal + noise
    filename = Path(path) / f'{name}.npy'
    np.save(filename, layer)
    if display:
        display_layer(layer)


def make_landscape_layers(path, L, scale, signal_range=(0, 10), noise_range=(0, 1), display=False):
    '''
    :param path: The directory in which to save the layer ndarrays.
    :param L: the size of the layer
    :param signal: all layers except wind direction have a signal value in the range [0, signal).
    :param scale: the smoothness of the perlin noise. higher is smoother.
    :param display: show each layer
    :return:
    '''
    '''
    :return:
    '''
    make_layer(path, 'topography', L=L, scale=scale, signal_range=signal_range, noise_range=noise_range, display=display)
    make_layer(path, 'temperature', L=L, scale=scale, signal_range=signal_range, noise_range=noise_range, display=display)
    make_layer(path, 'humidity', L=L, scale=scale, signal_range=signal_range, noise_range=noise_range, display=display)
    make_layer(path, 'wind_speed', L=L, scale=scale, signal_range=signal_range, noise_range=noise_range, display=display)
    make_layer(path, 'wind_direction', L=L, scale=scale, signal_range=(0, 2 * np.pi),
               noise_range=(-np.pi / 2, np.pi / 2), display=display)


def start_fire(landscape):
    '''
    Start a fire in the center of the landscape
    '''
    i = landscape.shape[0] // 2
    j = landscape.shape[1] // 2
    landscape[i, j, wf.L_FIRE] = 1
    landscape[i, j, wf.L_TREE] = 0
    return landscape


def make_initial_and_final_state(landscape_dir='data/synth_data/', max_time=20, seed=None, display=True):
    """
    Create a landscape, start an initial fire, simulate a burn for max_time time steps,
    and then save the initial fire and final fire landscapes.

    Parameters:
        :param landscape_path: directory containing landscape layer files
        :param seed: optional integer seed

    """
    max_time = int(max_time)  # in case it comes in as a string from the command line

    if seed is not None:
        np.random.seed(int(seed))

    landscape_path = Path(landscape_dir)
    landscape = wf.make_landscape_from_dir(landscape_path)

    landscape = start_fire(landscape)
    init_landscape = copy.deepcopy(landscape)

    # Simulate burn
    gamma = 0.7
    final_landscape, state_maps = wf.simulate_fire(landscape, max_time, wf.make_calc_prob_fire(gamma),
                                                   with_state_maps=True)

    # save initial and final landscapes
    np.save(landscape_path / 'init_landscape.npy', init_landscape)
    np.save(landscape_path / 'final_landscape.npy', final_landscape)
    if display:
        wf.show_landscape(init_landscape)
        wf.show_landscape(final_landscape)


def make_synthetic_dataset(parent_dir, n, L, f, scale, max_time, seed=None):
    try:
        f = int(f)
    except ValueError:
        pass

    # create a directory for this dataset based on the parameters
    dn = Path(parent_dir) / f'plain_n_{n}_L_{L}_f_{f}_scale_{scale}_max_time_{max_time}_seed_{seed}'
    print('dataset directory:', dn)

    # seed the RNG to reproduce the same landscapes
    if seed is not None:
        np.random.seed(seed)

    for i in range(n):
        # each landscape has its own directory
        padded_i = str(i).zfill(len(str(n)))  # e.g. n=100, i=11, padded_i='011'
        landscape_dir = dn / f'landscape_{padded_i}'
        landscape_dir.mkdir(parents=True, exist_ok=True)
        make_landscape_layers(landscape_dir, L=L, scale=scale, signal_range=(0, 10), noise_range=(0, 1))
        make_initial_and_final_state(landscape_dir, max_time, display=False)


def make_landscapes(dataset_dir, n, L, scale, seed=None, display=False):
    '''
    Make the layers of n landscapes, with each landscape in its own directory underneath dataset_dir.

    :param dataset_dir:
    :param n: the number of landscapes to make
    :param L: the size of the landscape
    :param f: the maximum of the minimum height of the layers
    :param scale: the variability in layer height
    :param int seed: use a seed for reproducibility
    :param display: if True, show every layer generated.
    :return:
    '''
    # reproducibility
    if seed is not None:
        np.random.seed(seed)

    dn = Path(dataset_dir)
    for i in range(n):
        # each landscape has its own directory
        padded_i = str(i).zfill(len(str(n)))  # e.g. n=100, i=11, padded_i='011'
        landscape_dir = dn / f'landscape_{padded_i}'
        landscape_dir.mkdir(parents=True, exist_ok=True)
        make_landscape_layers(landscape_dir, L=L, scale=scale,
                              signal_range=(0, 10), noise_range=(0, 1), display=display)


def burn_landscapes(dataset_dir=None, landscape_dir=None, max_time=20, num_periods=1, seed=None, display=False,
                    func_type='original'):
    # reproducibility
    if seed is not None:
        np.random.seed(seed)

    # burn either a single landscape or all the landscapes in a directory
    if (dataset_dir and landscape_dir) or (not dataset_dir and not landscape_dir):
        raise Exception('Please specify exactly one of dataset_dir or landscape_dir')
    elif landscape_dir:
        dirs = [Path(landscape_dir)]
    else:
        dirs = [d for d in Path(dataset_dir).iterdir() if d.is_dir()]

    def save_landscape(path, landscape, i):
        padded_i = str(i).zfill(len(str(num_periods)))  # e.g. num_periods=10, i=2, padded_i='02'
        np.save(path / f'landscape_burn_{padded_i}.npy', landscape)

    for path in dirs:
        landscape = wf.make_landscape_from_dir(path)

        # the initial burn
        landscape = start_fire(landscape)
        save_landscape(path, landscape, 0)
        if display:
            wf.show_landscape(landscape)

        if func_type == 'dz_logits':
            # spread driven by dz. biased against spreading downhill
            prob_func = features.make_logits_fire_func(features.make_scaled_logits_func(
                bias=-6, feature_scale=np.array([10.0, 0.0, 0.0, 0.0])
            ))
        elif func_type == 'wind_logits':
            # spread driven by wind speed and direction
            prob_func = features.make_logits_fire_func(features.make_scaled_logits_func(
                bias=-6, feature_scale=np.array([0.0, 0.0, 0.0, 3.0])
            ))
        elif func_type == 'balanced_logits':
            # NOTE, these weights are tuned to f=10, L=100, scale=400.
            # feature scale: dz_scale, temperature_scale, humidity_scale, wind_scale
            # spread driven by a balance of dz and wind
            # dz is scaled to show moderate influence from topographical gradients in the standard landscape
            # dz is scaled to have between ~1/8 and 8
            # temperature is scaled to have between 0 and ~2.2 logits
            # humidity is scaled to have between 0 and -2.2 logits
            # wind is scaled to have between 1/8 and 7-10 logits
            prob_func = features.make_logits_fire_func(features.make_scaled_logits_func(
                bias=-7, feature_scale=np.array([6.0, 0.2, 0.2, 0.8])
            ))
        elif func_type == 'original':
            # the original probability func
            prob_func = wf.make_calc_prob_fire(gamma=0.7)
        else:
            raise Exception('Unrecognized func_type', func_type)

        # burn the landscape sequentially num_periods times for max_time steps each burn.
        for i in range(1, num_periods+1):
            landscape = wf.simulate_fire(landscape, max_time, prob_func, with_state_maps=False)
            save_landscape(path, landscape, i)
            if display:
                wf.show_landscape(landscape)


def main(*args):
    for fn in ('topography.npy', 'wind_speed.npy', 'temperature.npy', 'humidity.npy'):
        path = Path('data/synth_data/') / fn
        layer = np.load(path)
        print(f'{fn} shape: {layer.shape}')
        display_layer(layer)


if __name__ == '__main__':
    main_parser = argparse.ArgumentParser()
    subparsers = main_parser.add_subparsers()

    parser = subparsers.add_parser('make_synthetic_dataset')
    parser.add_argument('--parent-dir', help='The directory in which to create the dataset directory')
    parser.add_argument('-n', type=int, help='The number of landscapes to create')
    parser.add_argument('-L', type=int, help='The length of the sides of the landscape grid')
    parser.add_argument('-f', type=float, help='The average value of the Perlin noise')
    parser.add_argument('--scale', type=int, help='The smoothness of the Perlin noise')
    parser.add_argument('--max-time', type=int, help='The length of time to simulate the fire')
    parser.add_argument('--seed', type=int, default=0,
                        help='Used to seed the random number generator for reproducibility')
    parser.set_defaults(func=make_synthetic_dataset)

    parser = subparsers.add_parser('make_initial_and_final_state')
    parser.add_argument('landscape_dir', help='directory containing landscape layers')
    parser.add_argument('max_time', type=int, help='The length of time to simulate the fire')
    parser.add_argument('--seed', type=int, default=0,
                        help='Used to seed the random number generator for reproducibility')
    parser.set_defaults(func=make_initial_and_final_state)

    parser = subparsers.add_parser('make_perlin_layer')
    parser.add_argument('-L', type=int, help='The length of the sides of the landscape grid')
    parser.add_argument('-f', type=float, help='The average value of the Perlin noise')
    parser.add_argument('--scale', type=int, help='The smoothness of the Perlin noise')
    parser.add_argument('--norm', default=False, action='store_true', help='Normalize layer values to be between 0 and 1')
    parser.add_argument('--display', default=False, action='store_true', help='Display layer')
    parser.set_defaults(func=make_perlin_layer)

    parser = subparsers.add_parser('make_landscapes')
    parser.add_argument('--dataset-dir', help='The directory in which to create the landscape directories')
    parser.add_argument('-n', type=int, help='The number of landscapes to create')
    parser.add_argument('-L', type=int, help='The length of the sides of the landscape grid')
    parser.add_argument('--scale', type=int, help='The smoothness of the Perlin noise')
    parser.add_argument('--seed', type=int, default=0,
                        help='Used to seed the random number generator for reproducibility')
    parser.add_argument('--display', default=False, action='store_true', help='Display each landscape layer')
    parser.set_defaults(func=make_landscapes)

    parser = subparsers.add_parser('burn_landscapes')
    parser.add_argument('--dataset-dir', help='directory containing landscape directories', default=None)
    parser.add_argument('--landscape-dir', help='directory containing landscape layers', default=None)
    parser.add_argument('--max-time', type=int, help='The length of time to simulate the fire during each period')
    parser.add_argument('--num-periods', type=int, help='The number of burns to simulate sequentially.', default=1)
    parser.add_argument('--seed', type=int, default=0,
                        help='Used to seed the random number generator for reproducibility')
    parser.add_argument('--display', default=False, action='store_true', help='Display each burned landscape')
    parser.add_argument('--func-type', default='original', choices=['original', 'balanced_logits', 'dz_logits',
                                                                    'wind_logits'])
    parser.set_defaults(func=burn_landscapes)

    parser = subparsers.add_parser('display_layers_from_dir', help='Display the layers in a landscape dir')
    parser.add_argument('--path', help='directory containing landscape layers')
    parser.set_defaults(func=display_layers_from_dir)

    parser = subparsers.add_parser('display_landscape_file', help='Display a landscape burn file')
    parser.add_argument('--path', help='npy file containing landscape')
    parser.set_defaults(func=display_landscape_file)

    parser = subparsers.add_parser('plot_landscape_histograms_from_file', help='Display a landscape burn file')
    parser.add_argument('--path', help='npy file containing landscape')
    parser.set_defaults(func=plot_landscape_histograms_from_file)



    args = main_parser.parse_args()
    # print(args)
    func = args.func
    kws = vars(args)
    del kws['func']
    print(kws)
    func(**kws)



