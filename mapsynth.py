"""
Create synthetic landscapes, including topography, humidity, temperature, wind speed and wind direction.

For each landscape, first a set of layers is generated (one layer for topography, for temperature, etc.).
Then the layers are stitched into a landscape (3-d ndarray), an initial burn is created, a wildfire
simulation is run, and then the initial burned landscape and final burned landscape are saved.

Usage

Test making a perlin noise layer:

python mapsynth.py make_perlin_layer -L50 -f10 --scale=100 --display --norm

Make 10 synthetic landscapes with initial and final burns:

python mapsynth.py make_synthetic_dataset --parent-dir=test_data -n10 -L50 -f10 --scale=100 --max-time=20 --seed=1

Make 2 landscapes, layers only:

python mapsynth.py make_landscapes --dataset-dir=test_data/test_dataset -n2 -L100 -f10 --scale=100 --seed=1 --display

Burn every landscape in a dataset for 4 * 20 timesteps, saving the landscape initially and after every 20 time step burn:

python mapsynth.py burn_landscapes --dataset-dir=test_data/test_dataset --max-time=20 --num-periods=4 --seed=1 --display

"""

import argparse
import copy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import wildfire as wf


def fade(t):
    """
        FADE: fade function to generate perlin noise_type

            ARGS:
                t: value
                mode:
    """

    return 6*t**5 - 15*t**4 + 10* t**3


def grad(site, neighbors):
    pass


def lerp(a, b, X):
    return a + X * (b - a)


class Landscape(object):

    def __init__(self, L=20):
        #square dimension of the landscape lattice:
        self.L = L

        #array  representing the lattice(fill it with trees):
        self.land = np.full((self.L, self.L), 2, dtype=int)

        #array representing the topography:
        self.top = np.zeros((self.L, self.L), dtype=float)

        #dictionary describing neighbors for each cell:
        self.neighbors = {}

        #ordered list of all indices:
        self.indices = np.array(list(np.ndindex(self.land.shape)))

        #indices of the sites on fire:
        self.on_fire = []

    def get_neighbors(self):
        """
            GET_NEIGHBORS: creates an dictionary of adjacency lists for each site on
                          the lattice:

                        ARGS:
                            NONE

                     RETURNS:
                            NONE initializes the dictionary of adjacency lists
        """
        #create a reference list to check if we have exhausted the row:
        row_len = [i for i in range(self.L)]

        for idx, site in enumerate(self.indices):

            #every site must have an adjacency list (set of 4 neighbors):
            adj = []

            #check neighbor directly above first:
            if site[0] - 1 in row_len:
                adj.append((site[0] - 1, site[1]))

            #next, check neighbor directly to the right:
            if site[1] + 1 in row_len:
                adj.append((site[0], site[1] + 1))

            #next, check neighbor directly below:
            if site[0] + 1 in row_len:
                adj.append((site[0] + 1, site[1]))

            #finally, check neighbor directly to the left:
            if site[1] - 1 in row_len:
                adj.append((site[0], site[1] - 1))

            #update the dictionary of adjacency lists:
            self.neighbors[tuple(site)] = adj
        return

    def initialize_fire(self, amt_spread):
        """
            INITIALIZE_FIRE: Randomly construct a connected component of sites
                             that have the value of 1. Do this by selecting a
                             coordinate on the lattice at random, then following
                             random edges of its neighbors to ignite (turn 1).
                             Continue following excess edges for the input
                             amount of spreading. Nearest neighbors to the
                             selected coordinate are always burned. This way,
                             a centroid is defined and the fire is not biased
                             in one direction.

                            ARGS:
                                amt_spreading: the number of edges to follow
                                               out from the point of outbreak.
                                               Type: int

                        RETURNS:
                                NONE, assigns the selected coordinates in
                                self.land to be in state 1.

        """

        #select a site to ignite:
        selection = np.random.choice(list(range(self.L**2)), p=None, size=1)

        #ignite site at coordinate associated with the seleciton (index):
        coord = (self.indices[selection][0][0], self.indices[selection][0][1])

        self.land[coord] = 1

        #propagate fire using random selections from the excess degree of coord:
        neighbors = self.neighbors[coord]

        #randomly ignite 2 of the nearest neighbors:
        init_ignite = np.random.choice(range(len(neighbors)), size=2)
        for n in neighbors:
            self.land[n] = 1

        #now, follow the desired number of excess edges and spread the fire:
        t = 0
        while t <= amt_spread:
            self.get_sites_on_fire()
            seen = set((i[0],i[1]) for i in self.on_fire)

            for num_edges in self.on_fire:
                temp = []
                for neigh in self.neighbors[num_edges[0], num_edges[1]]:
                    #generate a random number:
                    if neigh not in seen:
                        temp.append(neigh)

                #randomly choose a neighbor to ignite:
                try:
                    n2choosefrom = [i for i in range(len(temp))]

                    rn = np.random.choice(n2choosefrom)
                    seen.update([temp[rn]])
                except:
                    pass

            for i in seen:
                self.land[i] = 1
            t+=1
        return

    def get_sites_on_fire(self):
        """
            GET_SITES_ON_FIRE: gets the indices of sites that are assigned the
                               value 1 (on fire).

                               ARGS:
                                    None

                            RETURNS:
                                    an array of indices
        """
        #get the row and column indices for each of the sites on fire:
        i, j = np.where(self.land == 1)

        #zip them into a series of tuples:
        coords = list(zip(i,j))

        #update on_fire:
        self.on_fire = np.array(coords)

    def perlin_terrain(self, f, noise, scale=100, save=False):
        """
            PERLIN_TERRAIN: Generates ranodom 2D height maps that mimic smoothed
                            terrain.

                        ARGS:
                            f: a scale factor and the height displacement
                              (phase displacement shift/ translation). Type: int

                            noise: the noise function that defines the deformation
                                   of a point across the 2D space. Type: str
                                   Acceptible Input:
                                            'Sinosoid': generates a sinosoidal
                                                      surface centered at h+1.
                                                      (see: https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html)

                                            'Perlin': generates perlin noise
                                                      over the 2D space to
                                                      represent a height map.
                                                      requires a scale input,
                                                      which determines the
                                                      amount of smoothing
                                                      (i.e interpolation between
                                                      sites on the lattice)

                            scale: the amount of smoothing. Type: int

                            save: whether to save self.top as a numpy object after
                                  each time scale increment. (default = True)
                                  Type: boolean

                    RETURNS:
                            2D nparray describing a height map.
        """
        #NOTES:
        #octaves are the number of waves to combine
        #frequency is the number of oscillations per time step
        #amplitude is the height of any given point on the waveform
        #get row,col:

        if noise == 'Saddle':

            #initialize a meshgrid:
            X, Y = np.mgrid[-1:1:self.L+0j, -1:1:self.L+0j]

            #calculate heights at each site (centered about Z= f + 1):
            Z = ((f*(X + Y) * np.exp(-f*(X**2 + Y**2)) + 1) * f + f) + 1
            self.top = Z

        if noise == 'Perlin':

            #choose different intervals to pick samples from:
            low_high = np.array([(0, 1), (0.25, 0.75), (0.33, 0.67), (0.375, 0.625)])

            #at each site in the grid, initialize some noise
            for j in range(self.L):
                for i in range(self.L):
                    x = fade(np.random.uniform(low=0,high=1)) * f
                    y = fade(np.random.uniform(low=0,high=1)) * f

                    self.top[i, j] = x + y

            #how much zooming in or out we wish to do (smoothing as a function of time):
            t = 0
            while t <= scale:
                for site in self.indices:

                    #choose the interval to sample from:
                    lh = np.random.choice([0, 1, 2, 3])

                    #take a sample from the interval:
                    rn = np.random.uniform(low=low_high[lh][0],
                                           high=low_high[lh][1],
                                           size=len(self.neighbors[site[0], site[1]]))

                    #store the gradient interpolations:
                    lerps = np.zeros(len(self.neighbors[site[0], site[1]]))

                    # interpolate gradient given heights of sites neighbors:
                    for idx, n in enumerate(self.neighbors[site[0], site[1]]):
                        lerps[idx] = lerp(self.top[n], self.top[site[0], site[1]], rn[idx])

                    #average over all gradients and save as a height:
                    self.top[site[0], site[1]] = np.average(lerps)

                if save:
                    self.save_topography('data/landscapes/perlin_'+ str(self.L)+'_x_'+str(self.L)+'_'+str(t))

                #increment the time step:
                t += 1

    def save_topography(self, file_path):
        """
            SAVE_TOPOGRAPHY: saves the topography matrix as a numpy object

                           ARGS:
                                file_path: name of the npy file, path to
                                           topography matrix

                        RETURNS:
                                NONE, saves output to a destination
        """

        np.save(file_path, self.top)



    def load_topography(self, file_path):
        """
            LOAD_TOPOGRAPHY: loads a numpy array representing a topography
                             matrix.

                           ARGS:
                                file_path: nameof the npy file representing the
                                           topography matrix (.npy extension).
                                           Type: str
        """

        self.top = np.load(file_path)

    def display_topography(self):
        X, Y = np.mgrid[:self.L, :self.L]

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(1,1,1, projection='3d')
        surf = ax.plot_surface(X,Y,self.top,cmap='Greens')
        plt.show()


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


def perlin_terrain(L, f, scale=100):
    """
        PERLIN_TERRAIN: Generates ranodom 2D height maps that mimic smoothed
                        terrain.

                    ARGS:
                        f: a scale factor and the height displacement
                          (phase displacement shift/ translation). Type: int

                        noise: the noise function that defines the deformation
                               of a point across the 2D space. Type: str
                               Acceptible Input:
                                        'Sinosoid': generates a sinosoidal
                                                  surface centered at h+1.
                                                  (see: https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html)

                                        'Perlin': generates perlin noise
                                                  over the 2D space to
                                                  represent a height map.
                                                  requires a scale input,
                                                  which determines the
                                                  amount of smoothing
                                                  (i.e interpolation between
                                                  sites on the lattice)

                        scale: the amount of smoothing. Type: int

                        save: whether to save self.top as a numpy object after
                              each time scale increment. (default = True)
                              Type: boolean

                RETURNS:
                        2D nparray describing a height map.
    """
    #NOTES:
    #octaves are the number of waves to combine
    #frequency is the number of oscillations per time step
    #amplitude is the height of any given point on the waveform
    #get row,col:


    # choose different intervals to pick samples from:
    low_high = np.array([(0, 1), (0.25, 0.75), (0.33, 0.67), (0.375, 0.625)])

    # at each site in the grid, initialize some noise
    for j in range(self.L):
        for i in range(self.L):
            x = fade(np.random.uniform(low=0,high=1)) * f
            y = fade(np.random.uniform(low=0,high=1)) * f

            self.top[i, j] = x + y

    #how much zooming in or out we wish to do (smoothing as a function of time):
    t = 0
    while t <= scale:
        for site in self.indices:

            #choose the interval to sample from:
            lh = np.random.choice([0, 1, 2, 3])

            #take a sample from the interval:
            rn = np.random.uniform(low=low_high[lh][0],
                                   high=low_high[lh][1],
                                   size=len(self.neighbors[site[0], site[1]]))

            #store the gradient interpolations:
            lerps = np.zeros(len(self.neighbors[site[0], site[1]]))

            # interpolate gradient given heights of sites neighbors:
            for idx, n in enumerate(self.neighbors[site[0], site[1]]):
                lerps[idx] = lerp(self.top[n], self.top[site[0], site[1]], rn[idx])

            #average over all gradients and save as a height:
            self.top[site[0], site[1]] = np.average(lerps)

        if save:
            self.save_topography('data/landscapes/perlin_'+ str(self.L)+'_x_'+str(self.L)+'_'+str(t))

        #increment the time step:
        t += 1


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


def make_layer(path, name, L=50, f=10, scale=100, mean_lim=(0, 10), noise_range=1, center_noise=False, display=False):
    # ranges from 0 to noise_range
    noise_layer = make_perlin_layer(L, f, scale, norm=True) * noise_range
    if center_noise:
        # ranges from -noise_range/2 to noise_range/2
        noise_layer = noise_layer - (noise_range / 2)

    mean_min, mean_max = mean_lim
    mean = np.random.random() * (mean_max - mean_min) + mean_min
    layer = noise_layer + mean
    filename = Path(path) / f'{name}.npy'
    np.save(filename, layer)
    if display:
        display_layer(layer)


def make_landscape_layers(path, L, f, scale, display=False):
    '''
    :param path: The directory in which to save the layer ndarrays.
    :param L: the size of the layer
    :param f: most layers (excepting wind_direction) have a minimum value ranging from 0 to f.
    :param scale: the smoothness of the perlin noise. higher is smoother.
    :param display: show each layer
    :return:
    '''
    '''
    :return:
    '''
    make_layer(path, 'topography', L=L, f=f, scale=scale, mean_lim=(0, f), noise_range=1, center_noise=False,
               display=display)
    make_layer(path, 'temperature', L=L, f=f, scale=scale, mean_lim=(0, f), noise_range=1, center_noise=False,
               display=display)
    make_layer(path, 'humidity', L=L, f=f, scale=scale, mean_lim=(0, f), noise_range=1, center_noise=False,
               display=display)
    make_layer(path, 'wind_speed', L=L, f=f, scale=scale, mean_lim=(0, f), noise_range=1, center_noise=False,
               display=display)
    make_layer(path, 'wind_direction', L=L, f=f, scale=scale, mean_lim=(0, 2 * np.pi), noise_range=np.pi / 4, center_noise=True,
               display=display)

    # for name in ['topography', 'temperature', 'humidity']:
    #     layer = make_perlin_layer(L, f, scale)
    #     filename = Path(path) / f'{name}.npy'
    #     np.save(filename, layer)
    #     if display:
    #         display_layer(layer)


    # # wind speed is a random value in [0, f] + perlin noise in [0, wsv]
    # wind_speed_variability = 1  # arbitrary constant
    # layer = np.random.random() * f + make_perlin_layer(L, f, scale, norm=True) * wind_speed_variability
    # filename = Path(path) / f'wind_speed.npy'
    # np.save(filename, layer)
    # if display:
    #     display_layer(layer)
    #
    # # wind direction is a random value in [0, 2*pi] + perlin noise ranging from [-wdv, wdv]
    # wind_direction_variability = np.pi / 8  # arbitrary constant
    # layer = np.random.random() * 2 * np.pi
    # layer += make_perlin_layer(L, f, scale, norm=True) * 2 * wind_direction_variability - wind_direction_variability
    # filename = Path(path) / f'wind_direction.npy'
    # np.save(filename, layer)
    # if display:
    #     display_layer(layer)


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
        make_landscape_layers(landscape_dir, L, f, scale)
        make_initial_and_final_state(landscape_dir, max_time, display=False)


def make_landscapes(dataset_dir, n, L, f, scale, seed=None, display=False):
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
        make_landscape_layers(landscape_dir, L, f, scale, display=display)


def burn_landscapes(dataset_dir=None, landscape_dir=None, max_time=20, num_periods=1, seed=None, display=False):
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

        # burn the landscape sequentially num_periods times for max_time steps each burn.
        for i in range(1, num_periods+1):
            prob_func = wf.make_calc_prob_fire(gamma=0.7)
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
    parser.add_argument('-f', type=float, help='The average value of the Perlin noise')
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
    parser.set_defaults(func=burn_landscapes)

    parser = subparsers.add_parser('display_layers_from_dir', help='Display the layers in a landscape dir')
    parser.add_argument('path', help='directory containing landscape layers')
    parser.set_defaults(func=display_layers_from_dir)

    args = main_parser.parse_args()
    # print(args)
    func = args.func
    kws = vars(args)
    del kws['func']
    print(kws)
    func(**kws)



