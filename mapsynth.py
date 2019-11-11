"""
Create synthetic landscapes, including topography, wind, temperature, and precipitation.

For each landscape, first a set of layers is generated (one layer for topography, for temperature, etc.).
Then the layers are stitched into a landscape (3-d ndarray), an initial burn is created, a wildfire
simulation is run, and then the initial burned landscape and final burned landscape are saved.
"""

import argparse
import copy
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


def make_same_params(L=50, f=10, scale=100):
    '''
    Make a list of parameters, one for each layer, for use with make_layers. The L, f, scale
    params are the same for every layer.

    :param L:
    :param f:
    :param scale:
    :return:
    '''
    layers = ['topography', 'temperature', 'humidity', 'wind_north', 'wind_east']
    return [(name, L, f, scale) for name in layers]


def make_layers(path, params):
    '''
    For each tuple in params, make a layer. Each tuple is (name, L, f, scale),
    where name is like 'humidity' or 'temperature'. The L, f, and scale params
    are used to generate a perlin noise landscape which is saved to a file
    called {name}.npy inside the directory `path`.

    By using a list of params, it is possible for different layers to have
     different values for `f` or `scale`.

    :param path: The directory in which to save the layer ndarrays.
    :param params: a list of parameters, one set of parameters per layer.
    :return:
    '''
    for name, L, f, scale in params:
        a = Landscape(L=L)
        a.get_neighbors()
        a.perlin_terrain(f=f, noise='Perlin', scale=scale, save=False)
        filename = Path(path) / f'{name}.npy'
        filename.parent.mkdir(parents=True, exist_ok=True)
        a.save_topography(filename)
        # a.display_topography()


def make_plain_seeded_landscapes(dir_name, n, L=50, f=10, scale=100):
    n = int(n)
    L = int(L)
    try:
        f = int(f)
    except ValueError:
        f = float(f)
    scale = int(scale)

    dn = Path(dir_name)
    for i in range(n):
        np.random.seed(i)
        landscape_dir = dn / f'plain_seed_{i}_L_{L}_f_{f}_scale_{scale}'
        landscape_dir.mkdir(parents=True, exist_ok=True)
        params = make_same_params(L, f, scale)
        make_layers(landscape_dir, params)


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

    # START FIRE IN CENTER OF LANDSACPE
    i = landscape.shape[0] // 2
    j = landscape.shape[1] // 2
    landscape[i, j, wf.L_FIRE] = 1
    landscape[i, j, wf.L_TREE] = 0
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

    print('loss_function')
    print(wf.loss_function(wf.get_state_layer(final_landscape), wf.get_state_layer(final_landscape)))
    print(wf.loss_function(wf.get_state_layer(init_landscape), wf.get_state_layer(final_landscape)))
    print('iou_fitness')
    print(wf.iou_fitness(final_landscape, final_landscape))
    print(wf.iou_fitness(final_landscape, init_landscape))


def make_synthetic_dataset(parent_dir, n, L, f, scale, max_time, seed=None):
    try:
        f = int(f)
    except ValueError:
        pass

    # create a directory for this dataset based on the parameters
    dn = Path(parent_dir) / f'plain_n_{n}_L_{L}_f_{f}_scale_{scale}_max_time_{max_time}_seed_{seed}'

    # seed the RNG to reproduce the same landscapes
    if seed is not None:
        np.random.seed(seed)

    for i in range(n):
        # each landscape has its own directory
        padded_i = str(i).zfill(len(str(n)))  # e.g. n=100, i=11, padded_i='011'
        landscape_dir = dn / f'landscape_{padded_i}'
        landscape_dir.mkdir(parents=True, exist_ok=True)
        params = make_same_params(L, f, scale)
        make_layers(landscape_dir, params)
        make_initial_and_final_state(landscape_dir, max_time, display=False)


def main(*args):
    for fn in ('topography.npy', 'wind_speed.npy', 'temperature.npy', 'humidity.npy'):
        path = Path('data/synth_data/') / fn
        raster = np.load(path)
        print(f'{fn} shape: {raster.shape}')
        display_raster(raster)


if __name__ == '__main__':
    main_parser = argparse.ArgumentParser()
    subparsers = main_parser.add_subparsers()

    # add a parser for make_synthetic_dataset
    parser = subparsers.add_parser('make_synthetic_dataset')
    parser.add_argument('parent_dir', help='The directory in which to create the dataset directory')
    parser.add_argument('n', type=int, help='The number of landscapes to create')
    parser.add_argument('L', type=int, help='The length of the sides of the landscape grid')
    parser.add_argument('f', type=float, help='The average value of the Perlin noise')
    parser.add_argument('scale', type=int, help='The smoothness of the Perlin noise')
    parser.add_argument('max_time', type=int, help='The length of time to simulate the fire')
    parser.add_argument('--seed', type=int, default=0,
                        help='Used to seed the random number generator for reproducibility')
    parser.set_defaults(func=make_synthetic_dataset)

    # add a parser for make_initial_and_final_state
    parser = subparsers.add_parser('make_initial_and_final_state')
    parser.add_argument('landscape_dir', help='directory containing landscape layers')
    parser.add_argument('max_time', type=int, help='The length of time to simulate the fire')
    parser.add_argument('--seed', type=int, default=0,
                        help='Used to seed the random number generator for reproducibility')
    parser.set_defaults(func=make_initial_and_final_state)

    args = main_parser.parse_args()
    # print(args)
    func = args.func
    kws = vars(args)
    del kws['func']
    print(kws)
    func(**kws)

    # globals()[args.func](*args.args)

    # make 20 landscapes of size 50x50 with perlin noise. Avg height=10 and
    # scale/smoothness=100 (more smoothing makes landscapes less multi-modal
    # and with a range of values closer to f.
    # make_plain_seeded_landscapes('test_data/plain', n=20, L=50, f=10, scale=100)

    #
    # python mapsynth make_synthetic_dataset test_data 10 50 10 100 20 1