'''
wildfire.py contains code for creating a landscape and simulating a forest fire using a cellular automata.

A landscape is a ndarray containing a layer for every feature -- cell state, height, humidity, ...
'''

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.metrics import jaccard_score
import numpy as np
from pathlib import Path
import sys

# Cell States
S_FIRE = 1
S_TREE = 2

# Landscape Layers, analogous to channels in deep neural network image models
L_STATE = 0
L_Z = 1


def output_state_maps(z_vals, state_maps, dirname='gif_fire'):
    dn = Path(dirname)
    if not dn.exists():
        dn.mkdir(parents=True)

    for i, frame in enumerate(state_maps):
        fig, ax = plt.subplots(figsize=(15, 10))
        cmap = colors.ListedColormap(['red', 'green'])
        ax.matshow(frame, cmap=cmap)
        plt.contour(z_vals, colors="b")
        figname = "gif_fire/{}.png".format(i)
        plt.savefig(figname)
        plt.close(fig)


def calc_prob_fire(cell, neighbors, gamma):
    '''
    Return the probability that cell catches on fire.
    :param cell: ndarray of shape (n_features,) containing cell values
    :param neighbors: ndarray of shape (n_neighbors, n_features) containing
                      the landscape values of the neighbors of cell
    :param float gamma: value between 0,1 that serves as a weight for the
                        importance of height
    :return: the probability that the cell catches fire
    '''
    if cell[L_STATE] != S_TREE:
        return 0  # only trees catch fire
    else:
        # sum the difference in height between cell and fire neighbors
        dz_sum = np.sum(neighbors[neighbors[:, L_STATE] == S_FIRE, L_Z] - cell[L_Z])
        if dz_sum == 0:
            dz_sum = 1

        num_neighbors = neighbors.shape[0]
        prob = gamma + (1 - gamma) * dz_sum / (num_neighbors * 2)
        # assert prob <= 1
        # print(prob)
        return prob


def get_neighbors(landscape):
    '''
    Figure out the integer indices of the von Neumann neighborhood of every cell in the 2d landscape.
    Cells in corners only have 2 neighbors and cells on edges have on 3 neighbors. Construct a 2d array, where
    each element is the integer indices of the neighboring cells of the element.

    Usage:
        neighbors = get_neighbors(landscape)
        # each row contains the landscape values of a neighbor of the cell at (row, col).
        nbs = landscape[neighbors[row, col]]

    :param landscape: a 3d ndarray. The first two dims are row and column. The 3rd dim is cell data.
    :return: a 2d ndarray containing integer indices tuples. Each tuple of indices is the indices of the neighbors of that cell.
    '''
    # get the integer index of every row and column
    # shape: (2, nrows, ncols)
    # e.g. for a 2x2 landscape: idx = array([[[0, 0], [1, 1]], [[0, 1], [0, 1]]])
    # so element (0, 1, 1) contains the row index for the 2nd row and second column: 1
    idx = np.indices(landscape.shape[:2])
    # the neighbor to the north for the top row is padded to -1 to indicate no neighbor
    # otherwise the neighbor is the idx of the cell in the row above
    # e.g. array([[[-1, -1], [ 0,  0]], [[-1, -1], [ 0,  1]]])
    north = np.pad(idx[:, :-1, :], ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=-1)
    south = np.pad(idx[:, 1:, :], ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=-1)
    west = np.pad(idx[:, :, :-1], ((0, 0), (0, 0), (1, 0)), mode='constant', constant_values=-1)
    east = np.pad(idx[:, :, 1:], ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=-1)
    # smush all 4 neighbors together
    stacked = np.stack([north, east, south, west], axis=3)
    # convert the north, east, south, west neighbors into a tuple of row indices and column indices
    # suitable for advanced integer indexing
    neighbors = np.empty(landscape.shape[:2], dtype=object)
    for i in range(neighbors.shape[0]):
        for j in range(neighbors.shape[1]):
            # filter out out-of-bounds neighbors
            n_idx = (stacked[0, i, j][stacked[0, i, j] >= 0], stacked[1, i, j][stacked[1, i, j] >= 0])
            neighbors[i, j] = n_idx
    return neighbors


def simulate_fire(landscape, gamma, max_time, fire_func):
    """
    Simulate a fire spreading over time.
    Sample call:
        fire_init(landscape, .7, 5, 20)

    :param ndarray landscape: 3 dimensional array containing the values (axis 2)
                              at each position (axis 0 and 1).
    :param float gamma: value between 0,1 that serves as a weight for the
                            importance of height
    :param float z_max : maximum height in current landscape
    :param int max_time:     amount of time to run the simulation
    :return list state_maps: a state matrix at each time step from the simulation
    """
    # neighbors[i, j] contain the indices of the neighbors of cell i,j.
    neighbors = get_neighbors(landscape)

    # Store the initial state of the landscape
    state_maps = [landscape[:, :, L_STATE].copy()]

    # BEGIN FIRE PROPOGATION
    for t in range(max_time):

        # what cells are trees bordering fire?
        is_tree = landscape[:, :, L_STATE] == S_TREE
        is_fire = np.zeros((landscape.shape[0] + 2, landscape.shape[1] + 2), dtype=bool)
        is_fire[1:-1, 1:-1] = landscape[:, :, L_STATE] == S_FIRE
        is_fire_north = is_fire[:-2, 1:-1]
        is_fire_south = is_fire[2:, 1:-1]
        is_fire_east = is_fire[1:-1, 2:]
        is_fire_west = is_fire[1:-1, :-2]
        is_border = is_tree & (is_fire_north | is_fire_south | is_fire_east | is_fire_west)
        # indices as (row_idx, col_idx) tuple
        # e.g. (array([0, 0, 1, 2, 2]), array([1, 2, 0, 1, 2]))
        border_idx = np.nonzero(is_border)
        border_size = len(border_idx[0])

        # calculate spread probability for those locations
        border_probs = np.zeros(border_size)
        for i in range(border_size):
            row = border_idx[0][i]
            col = border_idx[1][i]
            border_probs[i] = fire_func(landscape[row, col], landscape[neighbors[row, col]], gamma)

        # spread fire
        border_fire = border_probs > np.random.random(border_size)
        landscape[border_idx[0][border_fire], border_idx[1][border_fire], L_STATE] = S_FIRE

        # record the current state of the landscape
        state_maps.append(landscape[:, :, L_STATE].copy())
    return state_maps


def make_landscape(landscape_filename):
    z_vals = np.load(landscape_filename)
    states = np.full(z_vals.shape, S_TREE, dtype=np.float)
    landscape = np.stack([states, z_vals], axis=2)
    return landscape


def loss_function(predicted, truth):
    """
    Calculate loss of the full simulationq based on the Jaccard Index
    
    Must not count non fire space towards correct classifications
    
    Sample call:
        loss_function(predicted, truth)
   
    :param predicted : the e nxn matrix representing the states of each cell in the landscape at the final iteration of the fire simulation.
    :param true : the matrix representing the ground truth states of each cell in the landscape
        
  
    :return loss : the loss of the resulting set of state classifications
        
    """
    
    # convert matrices to set of indeces, enumerate indeces, perform intersection over union on the two sets
    num_cells = predicted.size
    predicted_array = np.reshape(predicted,num_cells)
    true_array = np.reshape(truth,num_cells)
    
    # only consider the sites that have the possibility of catching fire
    fire_site_idxs = np.where(((true_array == 1) | (true_array == 2)))
    
    true_fires = true_array[fire_site_idxs]
    predicted_fires = predicted_array[fire_site_idxs]

    print(true_fires)
    print(predicted_fires)
    IoU = jaccard_score(true_fires, predicted_fires)
    
    return(IoU)
    

def main():
    # read landscape and simulation time
    landscape_file = sys.argv[1] # landscape raster file (height of land)
    time_steps = int(sys.argv[2]) # simulation time steps
    ground_truth = np.load(sys.argv[3]) # ground truth state raster
    
    z_vals = np.load(landscape_file)
    # make landscape
    landscape = make_landscape(landscape_file)

    # START FIRE IN CENTER OF LANDSACPE
    i = landscape.shape[0] // 2
    j = landscape.shape[1] // 2
    landscape[i, j, L_STATE] = S_FIRE

    # simulate a fire
    state_maps = simulate_fire(landscape, .7, time_steps, calc_prob_fire)

    # convert the landscape state over time to images and save.
    output_state_maps(z_vals, state_maps)
    
    # final predicted state_map
    predicted = state_maps[-1]
    # calculate loss
    loss = loss_function(predicted, ground_truth)

if __name__ == '__main__':
    main()





