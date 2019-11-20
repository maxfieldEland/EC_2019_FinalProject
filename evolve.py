"""
Run a GP wildfire experiment.

Inputs:
- Initial landscape file
- Final landscape file
- CA simulation time steps
- CA simulation reps. The simulation is run repeatedly to generate a probability distribution/heatmap indicating
  what proportion of reps a cell is on fire at the end of a simulation. This smooths over the stochasticity inherent
  in using a probabilistic CA state transition function.

"""

import argparse
from copy import deepcopy
from deap import base
from deap import creator
from deap import tools
from deap import gp
from deap import algorithms
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import operator
import random

from wildfire import L_Z, L_FIRE, L_TREE, L_HUM, L_TEMP, L_WS, L_WD
from wildfire import simulate_fire, iou_fitness,multi_sim_iou_fitness


def plot_tree(tree):
    # raise NotImplementedError()
    # Visualize
    nodes, edges, labels = gp.graph(tree)
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    # this is broken
    # pos = nx.graphviz_layout(g, prog="dot")
    pos = nx.drawing.layout.spiral_layout(g)
    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels)
    plt.title(str(tree))
    plt.show()


def protectedDiv(left, right):
    if (right != 0):
        return (left/right)

    else:
        return 1


def make_prob_fire(individual, toolbox):
    func = toolbox.compile(individual)
    # print(func)

    def prob_func(neighborhood):
        # ignore wind for now
        cell = neighborhood[0, :-2] # (fire, tree, z, temp, hum)
        neighbors = neighborhood[1:, :]
        mean_temp = neighborhood[:, L_TEMP].mean()
        mean_hum = neighborhood[:, L_HUM].mean()
        # mean_wn = neighborhood[:, L_WN].mean()
        # mean_we = neighborhood[:, L_WE].mean()
        delta_z = neighborhood[:, L_Z] - cell[L_Z]  # the vertical distance from cell to neighbor
        # fire burns uphill faster, so weight fires below cell more and fires above cell less.
        mean_weighted_fire = np.mean(neighborhood[:, L_FIRE] * np.exp(-delta_z))
        # the sum of the delta_z of neighbors that are on fire. I'm still trying to figure out what this means.
        dz_sum = np.sum(neighbors[neighbors[:, L_FIRE] == 1, L_Z] - cell[L_Z])
        args = (*cell, mean_temp, mean_hum, mean_weighted_fire, dz_sum)
        try:
            # print("\n\n")
            # print(func)
            # print(individual)
            # print("Cell", *cell)
            # print("Temp", mean_temp)
            # print("Hum", mean_hum)
            # print("MWF", mean_weighted_fire)
            # print("dz_sum", dz_sum)
            return func(*cell, mean_temp, mean_hum, mean_weighted_fire, dz_sum)
        except ValueError:
            # print("\n\n")
            print('func:', individual)
            print('func args:', args)
            raise

    return prob_func


def main(init_landscape_path, final_landscape_path, max_time):
    # questions:
    # What is the right range and distribution for constants? (No fancy constant optimization like in Eureka)
    # How is max_time chosen (co-evolved?)

    # Issue:
    # Wind and neighborhoods. B/c neighborhood omits out-of-bounds neighbors, it is impossible to say which neighbor in
    # the neighborhood corresponds to the north neighbor of a cell. So if the wind is blowing south, there is no
    # way to know that it is blowing the fire from the north southward.

    # Load data
    init_landscape = np.load(init_landscape_path)
    final_landscape = np.load(final_landscape_path)
    max_time = int(max_time)

    # Hyperparameters
    seed = 1
    eph_const_min = -10
    eph_const_max = 10
    eph_const_sigma = 1
    pop_size = 100  # 300

    # The primitives of the tree and their arities.
    pset = gp.PrimitiveSet("main", 9, prefix='x')
    pset.renameArguments(x0='fire', x1='tree', x2='z', x3='temp', x4='hum',
                         x5='mean_temp', x6='mean_hum', x7='mean_weighted_fire', x8='dz_sum')
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(math.sin, 1)
    # Issue: log fails when passed negative numbers.
    # pset.addPrimitive(math.log, 1)
    # These generate constants to be inserted into trees
    pset.addEphemeralConstant("randunif", lambda: random.random() * (eph_const_max - eph_const_min) + eph_const_min)
    pset.addEphemeralConstant("randnorm", lambda: np.random.randn() * eph_const_sigma)

    # Define an individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=pset)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    def evalCA(individual, init_landscape, final_landscape, max_time):

        prob_func = make_prob_fire(individual, toolbox)
        pred_landscape = simulate_fire(deepcopy(init_landscape), max_time, prob_func)
        fitness = iou_fitness(final_landscape, pred_landscape)
        return (fitness,)  # return a tuple

    toolbox.register("evaluate", evalCA, init_landscape=init_landscape,
                     final_landscape=final_landscape, max_time=max_time)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats, halloffame=hof, verbose=True)
    tree = gp.PrimitiveTree(hof[0])
    print(tree)
    # print(hof)
    # print(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('func', help='a function to run')
    parser.add_argument('args', metavar='ARG', nargs='*', help='any arguments for the function')
    args = parser.parse_args()
    globals()[args.func](*args.args)


def multi_sim_main(init_landscape_paths, final_landscape_paths, max_time):
    # Run simulation on a series of landscapes and penalize based on average IoU from each individual simulation
    # 

    num_simulations = len(init_landscape_paths)
    # Data structure to hold initial and final landscapes
    init_landscapes = np.ndarray(shape = (100,100,7,num_simulations))
    final_landscapes = np.ndarray(shape = (100,100,7,num_simulations))
    
    
    # Load all landscapes into init and final landscape structures
    for path_idx in range(len(init_landscape_paths)):
        init_landscapes[:,:,:,path_idx] = np.load(init_landscape_paths[path_idx])
        final_landscapes[:,:,:,path_idx] = np.load(final_landscape_paths[path_idx])
    max_time = int(max_time)

    # Hyperparameters
    seed = 1
    eph_const_min = -10
    eph_const_max = 10
    eph_const_sigma = 1
    pop_size = 100  # 300
    num_gens = 20

    # The primitives of the tree and their arities.
    pset = gp.PrimitiveSet("main", 9, prefix='x')
    pset.renameArguments(x0='fire', x1='tree', x2='z', x3='temp', x4='hum', x5='mean_temp', x6='mean_hum', x7='mean_weighted_fire',x8='wind_term')
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(math.sin, 1)
    # Issue: log fails when passed negative numbers.
    # pset.addPrimitive(math.log, 1)
    # These generate constants to be inserted into trees
    pset.addEphemeralConstant("randunif", lambda: random.random() * (eph_const_max - eph_const_min) + eph_const_min)
    pset.addEphemeralConstant("randnorm", lambda: np.random.randn() * eph_const_sigma)

    # Define an individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=pset)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    avg_fitnessess = []
    
    def evalCA_MultiSim(individual, init_landscapes, final_landscapes, max_time,num_simulations):
        
        pred_landscapes = deepcopy(final_landscapes)
        
        # loop over landscapes and run simulation on each
        for i in range(num_simulations):
            prob_func = make_prob_fire(individual, toolbox)
            init_landscape = deepcopy(init_landscapes[:,:,:,i])
            # write predicted landscape to data structure
            pred_landscapes[:,:,:,i] = simulate_fire(init_landscape, max_time, prob_func)
            
        fitness = multi_sim_iou_fitness(final_landscapes, pred_landscapes)
        avg_fitnessess.append(np.average(fitness))
        return (fitness,)  # return a tuple

    toolbox.register("evaluate", evalCA_MultiSim, init_landscapes=init_landscapes,
                         final_landscapes=final_landscapes, max_time=max_time, num_simulations = num_simulations)

    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, num_gens, stats=mstats, halloffame=hof, verbose=True)
    tree = gp.PrimitiveTree(hof[0])
    print(tree)
    # print(hof)
    
    best_solution = toolbox.compile(hof[0])
    return(best_solution,log,avg_fitnessess)
    
def multi_sim_main_rand(init_landscape_paths, final_landscape_paths, max_time):
    # Run simulation on a series of landscapes and penalize based on average IoU from each individual simulation
    # 

    num_simulations = len(init_landscape_paths)
    # Data structure to hold initial and final landscapes
    init_landscapes = np.ndarray(shape = (100,100,7,num_simulations))
    final_landscapes = np.ndarray(shape = (100,100,7,num_simulations))
    
    
    # Load all landscapes into init and final landscape structures
    for path_idx in range(len(init_landscape_paths)):
        init_landscapes[:,:,:,path_idx] = np.load(init_landscape_paths[path_idx])
        final_landscapes[:,:,:,path_idx] = np.load(final_landscape_paths[path_idx])
    max_time = int(max_time)

    # Hyperparameters
    seed = 1
    eph_const_min = -10
    eph_const_max = 10
    eph_const_sigma = 1
    pop_size = 1  # 300
    num_gens = 1

    # The primitives of the tree and their arities.
    pset = gp.PrimitiveSet("main", 9, prefix='x')
    pset.renameArguments(x0='fire', x1='tree', x2='z', x3='temp', x4='hum', x5='mean_temp', x6='mean_hum', x7='mean_weighted_fire',x8='wind_term')
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(math.sin, 1)
    # Issue: log fails when passed negative numbers.
    # pset.addPrimitive(math.log, 1)
    # These generate constants to be inserted into trees
    #pset.addEphemeralConstant("randunif", lambda: random.random() * (eph_const_max - eph_const_min) + eph_const_min)
    #pset.addEphemeralConstant("randnorm", lambda: np.random.randn() * eph_const_sigma)

    # Define an individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=pset)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    avg_fitnessess = []
    
    def evalCA_MultiSim(individual, init_landscapes, final_landscapes, max_time,num_simulations):
        
        pred_landscapes = deepcopy(final_landscapes)
        
        # loop over landscapes and run simulation on each
        for i in range(num_simulations):
            prob_func = make_prob_fire(individual, toolbox)
            init_landscape = deepcopy(init_landscapes[:,:,:,i])
            # write predicted landscape to data structure
            pred_landscapes[:,:,:,i] = simulate_fire(init_landscape, max_time, prob_func)
            
        fitness = multi_sim_iou_fitness(final_landscapes, pred_landscapes)
        avg_fitnessess.append(np.average(fitness))
        return (fitness,)  # return a tuple

    toolbox.register("evaluate", evalCA_MultiSim, init_landscapes=init_landscapes,
                         final_landscapes=final_landscapes, max_time=max_time, num_simulations = num_simulations)

    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, num_gens, stats=mstats, halloffame=hof, verbose=True)
    tree = gp.PrimitiveTree(hof[0])
    print(tree)
    # print(hof)
    
    best_solution = toolbox.compile(hof[0])
    return(best_solution,log,avg_fitnessess)