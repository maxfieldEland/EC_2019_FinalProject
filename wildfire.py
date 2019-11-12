"""
wildfire.py contains code for creating a landscape and simulating a forest fire using a cellular automata.

A landscape is a ndarray containing a layer for every feature -- cell state, height, humidity, ...
"""

import sys
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.metrics import jaccard_score


# Cell States
S_FIRE = 1
S_TREE = 2

# Landscape Layers, analogous to channels in deep neural network image models
# CA state is one-hot encoded (one layer per state)
L_FIRE = 0  # fire state
L_TREE = 1  # tree state
L_Z = 2  # elevation
L_TEMP = 3  # temperature
L_HUM = 4  # humidity
L_WN = 5  # north component of wind speed vector
L_WE = 6  # east component of wind speed vector


def show_landscape(landscape):
    z = landscape[:, :, L_Z]
    state = get_state_layer(landscape)
    fig, ax = plt.subplots(figsize=(15, 10))
    cmap = colors.ListedColormap(['red', 'green'])
    ax.matshow(state, cmap=cmap)
    plt.contour(z, colors="b")
    plt.show()


def get_state_layer(landscape):
    '''
    State (tree, fire, etc.) is one-hot encoded as layers in landscape. Decode to a single
    layer where state is represented as a categorical variable using integer values.
    :param landscape:
    :return: a matrix whose elements are the state of each cell
    '''
    state = np.zeros(landscape.shape[:2])
    state[landscape[:, :, L_TREE] == 1] = S_TREE
    state[landscape[:, :, L_FIRE] == 1] = S_FIRE  # fire trumps trees
    return state


def make_landscape_from_dir(dn):
    dn = Path(dn)
    z = np.load(dn / 'topography.npy')
    temp = np.load(dn / 'temperature.npy')
    hum = np.load(dn / 'humidity.npy')
    wind_north = np.load(dn / 'wind_north.npy')
    wind_east = np.load(dn / 'wind_east.npy')
    tree = np.full(z.shape, 1.0, dtype=np.float)
    fire = np.zeros(z.shape)
    landscape = np.stack([fire, tree, z, temp, hum, wind_north, wind_east], axis=2)
    return landscape


def make_calc_prob_fire(gamma):
    """
    Parameters:
        :param float gamma: value between 0,1 that 
        serves as a weight for the importance of height
    """
    def prob_func(neighborhood):

        """
        Return the probability that cell catches on fire.

        Parameters:
            :param neighborhood: ndarray of shape (n_neighborhood, n_features) containing
            the landscape values of the cells of the neighborhood.

        Returns:
            :return: the probability that the cell catches fire
        """
        #  Neighborhood rows, in order, are (cell, north, east, south, west)
        #  Cells on the edges might not have all 4 neighbors.
        cell = neighborhood[0, :]
        neighbors = neighborhood[1:, :]

        if cell[L_TREE] != 1:
            return 0  # only trees catch fire
        else:
            # sum the difference in height between cell and fire neighbors
            dz_sum = np.sum(neighbors[neighbors[:, L_FIRE] == 1, L_Z] - cell[L_Z])
            if dz_sum == 0:
                dz_sum = 1

            num_neighbors = neighbors.shape[0]
            prob = gamma + (1 - gamma) * dz_sum / (num_neighbors * 2)
            return prob

    return prob_func


def get_neighborhoods(landscape):
    """
    Figure out the integer indices of the von Neumann neighborhood of every cell in the landscape. The indices of
    the neighborhood are computed naively, so cells on the edge of the landscape will have neighbors with indices that
    are off the edge of the landscape. E.g. the indices of the cell to the north of cell (0, 0) is (-1, 0).

    Usage:
        neighborhoods = get_neighborhoods(landscape)
        # each row contains the landscape values of a cell in the neighborhood at (row, col).
        nbs = landscape[tuple(neighborhoods[row, col])]

    Parameters:
        :param landscape: a 3d ndarray. The first two dims are row and column. The 3rd dim is cell data.

    Returns:
       a 4d array of shape (n_row, n_col, 2, 5). The first 2 axes correspond to the rows and
     columns of the landscape. The next 2 axis contain the row and column indices of the neighbors of the cell.
     The neighbors are ordered along axis 3 as (center, north, east, south, west).
    """
    # get the integer index of every row and column
    # shape: (2, nrows, ncols), where the first element of axis 0 contains the row indices
    # and the second element of axis 0 contains the column indices.
    # e.g. for a 3x3 landscape: idx = array(
    # [[[0, 0, 0],
    #   [1, 1, 1],
    #   [2, 2, 2]],
    #  [[0, 1, 2],
    #   [0, 1, 2],
    #   [0, 1, 2]]])
    # so element (0, 1, 1) contains the row index for the 2nd row and 2nd column: 1
    idx = np.indices(landscape.shape[:2])

    north = idx.copy()
    north[0] = idx[0] - 1  # north is cell in the row above
    east = idx.copy()
    east[1] = idx[1] + 1  # east is the cell in the column to the right
    south = idx.copy()
    south[0] = idx[0] + 1  # south is the cell in the row below
    west = idx.copy()
    west[1] = idx[1] - 1  # west is the cell in the column to the left
    stacked = np.stack([idx, north, east, south, west], axis=3)  # axes: (row/col, n_row, n_col, n_neighborhood)

    neighborhoods = np.transpose(stacked, (1, 2, 0, 3))  # axes (n_row, n_col, row/col, n_neighborhood)
    return neighborhoods


def simulate_fire(landscape, max_time, fire_func, with_state_maps=False):
    """
    Simulate a fire spreading over time.
    Sample call:
        simulate_fire(landscape, 20, make_calc_prob_fire(0.7))

    Parameters:
        :param ndarray landscape: 3 dimensional array containing the values (axis 2)
                                  at each position (axis 0 and 1).
        :param int max_time:     amount of time to run the simulation
        :param with_state_maps: if True, return (landscape, state_maps). Otherwise return only the landscape.

    Returns:
        :return: the final landscape, or the final landscape and a list of state maps.
    """
    # add padding so each cell in landscape has 4 neighbors, even cells on the edges of the landscape.
    padded_landscape = np.pad(landscape, pad_width=((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=np.nan)

    # tuple(neighbors[i, j]) contains the indices of the neighborhood of cell (i, j).
    neighborhoods = get_neighborhoods(padded_landscape)

    # Store the initial state of the landscape
    if with_state_maps:
        state_maps = [get_state_layer(padded_landscape[1:-1, 1:-1])]

    # BEGIN FIRE PROPOGATION
    for t in range(max_time):

        # what cells are trees that are not on fire but are bordering fire?
        is_tree = padded_landscape[:, :, L_TREE] == 1
        is_fire = padded_landscape[:, :, L_FIRE] == 1

        is_fire_padded = np.pad(is_fire, 1, mode='constant', constant_values=False)
        is_fire_north = is_fire_padded[:-2, 1:-1]  # a fire is north of cell (i, j) if cell (i-1, j) is on fire
        is_fire_south = is_fire_padded[2:, 1:-1]
        is_fire_east = is_fire_padded[1:-1, 2:]
        is_fire_west = is_fire_padded[1:-1, :-2]
        is_border = is_tree & np.logical_not(is_fire) & (is_fire_north | is_fire_south | is_fire_east | is_fire_west)
        border_idx = np.nonzero(is_border)
        border_size = len(border_idx[0])

        # calculate spread probability for those locations
        border_probs = np.zeros(border_size)
        for i in range(border_size):
            row = border_idx[0][i]
            col = border_idx[1][i]
            hood = padded_landscape[tuple(neighborhoods[row, col])]
            # print(t, row, col, hood)  # confirm that neighborhoods of cells on landscape edges contain "nan" cells
            border_probs[i] = fire_func(hood)

        # spread fire
        border_fire = border_probs > np.random.random(border_size)
        padded_landscape[border_idx[0][border_fire], border_idx[1][border_fire], L_FIRE] = 1
        padded_landscape[border_idx[0][border_fire], border_idx[1][border_fire], L_TREE] = 0

        # record the current state of the landscape
        if with_state_maps:
            state_maps.append(get_state_layer(padded_landscape[1:-1, 1:-1]))

    if with_state_maps:
        return padded_landscape[1:-1, 1:-1], state_maps
    else:
        return padded_landscape[1:-1, 1:-1]


def iou_fitness(true_landscape, pred_landscape):
    """
    Note: This function returns the same results as `loss_function`, taking landscapes instead of
    state maps to avoid the overhead of creating state maps from landscapes.

    Compute the intersection over union of the true landscape and predicted landscape.
    Perfect agreement is 1. Complete disagreement is 0.

    Parameters:
        :param true_landscape:
        :param pred_landscape:

    Returns:
        :return: the IoU, a float between 0 and 1.
    """

    # only consider sites that can burn (ignore rock, water)
    trees_or_fire_idx = np.nonzero((true_landscape[:, :, L_FIRE] == 1) | (true_landscape[:, :, L_TREE] == 1))
    y_true = true_landscape[:, :, L_FIRE][trees_or_fire_idx]
    y_pred = pred_landscape[:, :, L_FIRE][trees_or_fire_idx]
    iou = jaccard_score(y_true, y_pred)
    return iou


def loss_function(predicted, truth):
    """
    Calculate loss of the full simulation based on the Jaccard Index
    
    Must not count non fire space towards correct classifications
    
    Sample call:
        loss_function(predicted, truth)
   
    Parameters:
        :param predicted : the e nxn matrix representing the states of each 
        cell in the landscape at the final iteration of the fire simulation.

        :param true : the matrix representing the ground truth states of each cell in the landscape
        
    Returns:
        :return loss : the loss of the resulting set of state classifications
        
    """
    
    # convert matrices to set of indices, enumerate indices, perform intersection over union on the two sets
    num_cells = predicted.size
    predicted_array = np.reshape(predicted, num_cells)
    true_array = np.reshape(truth, num_cells)
    
    # only consider the sites that have the possibility of catching fire
    fire_site_idxs = np.where(((true_array == 1) | (true_array == 2)))
    
    true_fires = true_array[fire_site_idxs]
    predicted_fires = predicted_array[fire_site_idxs]

    print(true_fires)
    print(predicted_fires)
    IoU = jaccard_score(true_fires, predicted_fires)
    
    return IoU
    

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


def main():
    """
    Simulate a burn starting from the initial landscape, output the state maps, and calculate the loss of
    the simulated burned landscape compared to the final landscape.
    Usage: python wildfire.py data/synthetic_example/init_landscape.npy data/synthetic_example/final_landscape.npy 20
    """

    init_landscape_path = sys.argv[1]
    final_landscape_path = sys.argv[2]
    max_time = int(sys.argv[3])
    init_landscape = np.load(init_landscape_path)
    final_landscape = np.load(final_landscape_path)
    # Simulate burn
    gamma = 0.7
    pred_landscape, state_maps = simulate_fire(
        init_landscape, max_time, make_calc_prob_fire(gamma), with_state_maps=True)
    y_true = get_state_layer(final_landscape)
    y_pred = get_state_layer(pred_landscape)
    loss = loss_function(y_pred, y_true)
    print('loss:', loss)
    iou = iou_fitness(final_landscape, pred_landscape)
    print('iou fitness:', iou)
    show_landscape(final_landscape)
    show_landscape(pred_landscape)
    # convert the landscape state over time to images and save.
    output_state_maps(final_landscape[:, :, L_Z], state_maps)

if __name__ == '__main__':
    main()