'''
Simple null model.

The probability of a border cell transitioning to fire = p.
'''


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
from pathlib import Path
import random

from wildfire import L_Z, L_FIRE, L_TREE, L_HUM, L_TEMP, L_WS, L_WD
from wildfire import simulate_fire, iou_fitness


def main(landscape_dir, max_time, seed):
    pop_size = 10
    n_gen = 10
    cxpb = 0.0  # no crossover
    mutpb = 1.0  # all mutation
    n_sim = 4  # the number of simulated burns to average fitness over.
    np.random.seed(seed)

    burn_paths = list(sorted(Path(landscape_dir).glob('landscape_burn_*.npy')))
    # print(burn_paths)
    init_landscape = np.load(burn_paths[0])
    final_landscape = np.load(burn_paths[1])

    # Define an individual and population
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        prob = individual[0]

        def prob_func(neighborhood):
            return prob

        # average fitness across multiple simulations
        fits = np.zeros(n_sim)
        for i in range(n_sim):
            pred_landscape = simulate_fire(deepcopy(init_landscape), max_time, prob_func)
            fits[i] = iou_fitness(final_landscape, pred_landscape)

        return fits.mean(),   # return a tuple

    def mutate(individual):
        individual[0] = np.random.random()
        return individual,

    # algorithm params
    toolbox.register('mate', tools.cxUniform, indpb=1.0)
    toolbox.register("evaluate", evaluate)
    toolbox.register("select", tools.selRandom)
    toolbox.register("mutate", mutate)

    # statistics
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    pop, log = algorithms.eaSimple(population=pop, toolbox=toolbox, cxpb=cxpb, mutpb=mutpb, ngen=n_gen,
                                   stats=mstats, halloffame=hof, verbose=True)
    # print(pop)
    print(hof)
    # print(log)


if __name__ == '__main__':

    main_parser = argparse.ArgumentParser()
    subparsers = main_parser.add_subparsers()

    parser = subparsers.add_parser('main')
    parser.add_argument('--landscape-dir', help='directory containing landscape layers', default=None)
    parser.add_argument('--max-time', type=int, default=20,
                        help='The length of time to simulate the fire during each period')
    parser.add_argument('--seed', type=int, default=0,
                        help='Used to seed the random number generator for reproducibility')
    parser.set_defaults(func=main)

    args = main_parser.parse_args()
    func = args.func
    kws = vars(args)
    del kws['func']
    func(**kws)
