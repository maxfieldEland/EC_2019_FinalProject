'''
Methods:

Create A New Dataset

Create the landscape layers:
rm -rf ./foo && time python mapsynth.py make_landscapes --dataset-dir=foo -n2 -L100 --scale=400 --seed=1

Burn each landscape 4 times sequentially:

time python mapsynth.py burn_landscapes --dataset-dir=foo --max-time=20 --num-periods=4 --seed=1 --func-type balanced_logits

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
from sklearn.model_selection import train_test_split

import features
import wildfire
import mapsynth
from models import GeneticProgrammingModel, ConstantModel, WindModel, LogitsModel, WindLogitsModel, DzLogitsModel, BalancedLogitsModel


def make_evaluate_func(n_sim=1, max_time=20):
    '''
    Model expects an evaluate function with the signature `(x, y, fire_func)`.
    This function makes that function from `evaluate(x, y, fire_func, n_sim, max_time)`,
    and returns the function.
    :return:
    '''
    def evaluate_func(x, y, fire_func):
        # average fitness across multiple landscapes and multiple simulations
        return wildfire.evaluate(x, y, fire_func, n_sim=n_sim, max_time=max_time)

    return evaluate_func


def run_experiment(dataset_dir=None, landscape_dir=None, func_type='balanced_logits', n_rep=1, n_sim=1, max_time=20, seed=None):

    # Reproducibility first!
    np.random.seed(seed)
    random.seed(seed + 1)

    # load dataset, shape = n_land, n_row, n_col, n_feat
    init_landscapes = mapsynth.load_dataset_burn_i(0, dataset_dir=dataset_dir, landscape_dir=landscape_dir, func_type=func_type)
    final_landscapes = mapsynth.load_dataset_burn_i(1, dataset_dir=dataset_dir, landscape_dir=landscape_dir, func_type=func_type)

    # train-test split
    test_size = 10  # 10 train, 10 test
    x_train, x_test, y_train, y_test = train_test_split(init_landscapes, final_landscapes, test_size=test_size)
    print('x_train.shape, x_test.shape, y_train.shape, y_test.shape')
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # model = GeneticProgrammingModel(n_sim=n_sim, max_time=max_time)
    # model = ConstantModel(n_sim=n_sim, max_time=max_time)
    # model = WindModel(n_sim=n_sim, max_time=max_time)
    model = LogitsModel(n_sim=n_sim, max_time=max_time)
    # model = BalancedLogitsModel(max_time=max_time)
    # model = WindLogitsModel(max_time=max_time)
    # model = DzLogitsModel(max_time=max_time)

    for i_rep in range(n_rep):
        print('i_rep', i_rep)

        # train model
        model.fit(x_train, y_train)

        # evaluate model
        pred_train = model.predict(x_train)
        train_fitness = wildfire.multi_sim_iou_fitness(y_train, pred_train)
        pred_test = model.predict(x_test)
        test_fitness = wildfire.multi_sim_iou_fitness(y_test, pred_test)
        print(f'train_fitness {train_fitness}')
        print(f'test_fitness {test_fitness}')

        # plot
        plt.plot(np.arange(len(model.fitnesses)), model.fitnesses)
        plt.title('model fitness vs function evaluations')
        plt.ylabel('fitness')
        plt.xlabel('# function evaluations')
        plt.show()


def main(dataset_dir=None, landscape_dir=None, seed=None):
    run_experiment(dataset_dir=dataset_dir, landscape_dir=landscape_dir, seed=seed)


if __name__ == '__main__':

    main_parser = argparse.ArgumentParser()
    subparsers = main_parser.add_subparsers()

    parser = subparsers.add_parser('main')
    parser.add_argument('--dataset-dir', help='directory containing landscape directories', default=None)
    parser.add_argument('--landscape-dir', help='directory containing landscape layers', default=None)
    # parser.add_argument('--max-time', type=int, default=20,
    #                     help='The length of time to simulate the fire during each period')
    parser.add_argument('--seed', type=int, default=0,
                        help='Used to seed the random number generator for reproducibility')
    parser.set_defaults(func=main)

    args = main_parser.parse_args()
    func = args.func
    kws = vars(args)
    del kws['func']
    func(**kws)
