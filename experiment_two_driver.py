"""
Experiment 2 driver: should implement the same ideas found in 
the first experimental driver, but accounting for multiple
timesteps for each burn of the landscape.
"""

# Import universally used tools
import math
import features
import numpy as np 
import spread_functions as sp 
import matplotlib.pyplot as plt
from wildfire import iou_fitness
from mapsynth import make_landscapes 
from mapsynth import burn_landscapes 
from mapsynth import load_dataset_burns


# Import EXP2 specific functions:
from evolve import multi_step_main, multi_step_main_rand

#----------------------#
### Helper functions ###
#----------------------#

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def make_fitness_boxplot(fitnessess):
    fig = plt.figure(1, figsize=(9, 6))

    # Create an axes instance
    ax = fig.add_subplot(111)
    
    # Create the boxplot
    
    
    bp = ax.boxplot(fitnessess, patch_artist=True)
    
    ## change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set( color='#7570b3', linewidth=2)
        # change fill color
        box.set( facecolor = '#1b9e77' )
    
    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)
    
    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)
    
    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)
    
    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)
        # Save the figure
        
    ax.set_xticklabels(['Random Expression', 'Best Expression','Wind Logit','Elevation Logit','Balanced Logit'])
    plt.title("Fitness Distributions")
    fig.savefig('boxplot_fitnessess.png', bbox_inches='tight')


#----------------------------#
### Experimental functions ###
#----------------------------#


def experiment_two_driver(n, L, f, scale, max_time, num_periods, dspath,
    base_spread_func, base_func_type):
    """
    Driver file for experiment 2. Somewhat adapted version of Max's exp 1 
    driver file.

    Parameters:
        :int n:         Number of simulations (landscapes)
        :int L:         Side of landscape.
        :float f:       Mean of noise distribution
        :int scale:     Number of perlin iterations in landscape generation.
        :int max_time:  Amount of time to run a simulation for.
        :int num_periods: How many burn states every max_time gens to save.
        :str dspath:    Where to write the landscape dataset to.

        :func base_spread_func:     Function to burn landscapes with.
        :func base_func_type:       Logits function type passed to burn.

    Returns:
        tbd.
    """
   
    # Generate landscapes and their burns.
    make_landscapes(dataset_dir=dspath, n=n, L=L, scale=scale)
    burn_landscapes(
        spread_prob_func=base_spread_func, 
        dataset_dir=dspath, 
        max_time = max_time, 
        num_periods=num_periods, 
        func_type=base_func_type)

    #-------------------------------------------------------------#
    ### Train both the GP and the null model on these datasets. ###
    #-------------------------------------------------------------#

    best_ind, log, fitnessess = multi_step_main(dspath)
    rand_ind, rand_log, rand_fitnessess = multi_step_main_rand(dspath)

    #----------------------------#
    ### Report training stats: ###
    #----------------------------#

    # Find number of evaluations and corresponding fitness at each # of evals.
    numevals = []
    for i in log:
        numevals.append(i['nevals'])
        
    avg_fitness = []
    start_idx = 0 
    
    cumulative_func_evals = []

    for idx,k in enumerate(numevals):
        avg_fitness.append(np.average(fitnessess[start_idx:k+start_idx]))
        start_idx = start_idx + k
        cumulative_func_evals.append(start_idx)
    
    plt.figure(figsize = (10,10))
    plt.plot(numevals,avg_fitness)
    plt.title("Number of Evals vs. Avg Generational Fitness")
    plt.xlabel("Number of Evaluations")
    plt.ylabel("Avg. Fitness")
    plt.show()
    
    plt.figure(figsize = (10,10))
    plt.plot(cumulative_func_evals,avg_fitness)
    plt.title("Cummulative Number of Functiona Evals. Avg Generational Fitness")
    plt.xlabel("Number of Evaluations")
    plt.ylabel("Avg. Fitness")
    plt.show()

    #--------------------------------------------------#
    ###  Test trained models on newly generated data ###
    #--------------------------------------------------#

    # Generate new data in a different location for testing, using same params.
    dspath_test = 'data/test'
    make_landscapes(dspath_test, n, L, scale, seed=None, display=False)

    # For our evolved model, the random model, and each logit, create burns:
    burnparams = [(best_ind, 'best'), (rand_ind, 'rand'),
     (sp.spread_uphill_only, 'balanced_logit'), 
     (sp.spread_uphill_only, 'dz_logits'),
     (sp.spread_uphill_only, 'wind_logits')]

    for i in burnparams:
        burn_landscapes(spread_prob_func = i[0],
            dataset_dir = dspath_test,
            max_time = max_time, 
            func_type = i[1])

    # Test fitnesses for each of the above param combos:
    fits = {i[1]: [] for i in burnparams}

    for sim in range(n):

        # Load paths of each burn of each type:
        scapes = {i[1]:load_dataset_burns(dataset_dir=dspath_test,
            func_type = i[1]) for i in burnparams}

        for i in scapes.keys():
            print(i, len(scapes[i]))

        # Choose the last (?) of each of these
        predictions = {i:scapes[i][-1] for i in scapes.keys()}

        # Let ground truth be balanced landscape:
        truth = predictions['balanced_logit']

        # Append fitnesses
        for i in predictions.keys():
            fits[i].append(iou_fitness(truth, predictions[i]))


    # Produce figures
    make_fitness_boxplot(fits.values())


def main():
# define path and params
    dspath = 'data'
    n = 1
    L = 100
    f = 10
    scale = 500
    max_time = 1
    num_periods = 100
    base_spread_func = sp.spread_uphill_only
    base_func_type='balanced_logit'
    
    experiment_two_driver(
        n, L, f, scale, max_time, num_periods, dspath,
    base_spread_func, base_func_type)
     
main()
