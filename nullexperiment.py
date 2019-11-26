'''
Animate a model from a results directory:

    time python nullexperiment.py animate_results_model --landscape-dir=ds_200/landscape_0 \
    --results-dir=results/grid_search_exp_dataset1 --i-rep=0 --display --model-type=constant \
    --n-sim=10 --max-time=200

Test reading results of hyperparameter experiment:

time python nullexperiment.py plot_hyperparam_search_results --results-dir=results/grid_search_exp_dataset1 --i-rep=0 --model-type=gp

Test reading results of experiment

time python nullexperiment.py plot_results_model --results-dir=results/grid_search_exp_dataset1 --i-rep=0 --model-type=gp


'''

import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pickle
from pprint import pprint
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

import wildfire
import mapsynth
from models import (GeneticProgrammingModel, ConstantModel, WindModel, LogisticModel, BalancedLogitsModel,
                    DzLogitsModel, WindLogitsModel)


def get_model(model_type, n_sim=1, max_time=20, pop_size=100, n_gen=100, cxpb=0.5, mutpb=0.5):
    if model_type == 'gp':
        model = GeneticProgrammingModel(n_sim=n_sim, max_time=max_time, pop_size=pop_size, n_gen=n_gen, cxpb=cxpb, mutpb=mutpb)
    elif model_type == 'constant':
        model = ConstantModel(n_sim=n_sim, max_time=max_time, pop_size=pop_size, n_gen=n_gen, cxpb=cxpb, mutpb=mutpb)
    elif model_type == 'wind':
        model = WindModel(n_sim=n_sim, max_time=max_time, pop_size=pop_size, n_gen=n_gen, cxpb=cxpb, mutpb=mutpb)
    elif model_type == 'logistic':
        model = LogisticModel(n_sim=n_sim, max_time=max_time, pop_size=pop_size, n_gen=n_gen, cxpb=cxpb, mutpb=mutpb)
    elif model_type == 'balanced_logits':
        model = BalancedLogitsModel(max_time=max_time)
    elif model_type == 'wind_logits':
        model = WindLogitsModel(max_time=max_time)
    elif model_type == 'dz_logits':
        model = DzLogitsModel(max_time=max_time)
    else:
        raise Exception('Unrecognized model_type.', model_type)

    return model


def evaluate_model_on_final_and_future(model, x_train, x_test, y_train, y_test, future_train, future_test):
    # evaluate model
    pred_train = model.predict(x_train)
    train_fitness = wildfire.multi_sim_iou_fitness(y_train, pred_train)
    pred_test = model.predict(x_test)
    test_fitness = wildfire.multi_sim_iou_fitness(y_test, pred_test)
    print(f'train_fitness {train_fitness}')
    print(f'test_fitness {test_fitness}')

    # evaluate the future
    pred_future_train = model.predict(y_train)
    future_train_fitness = wildfire.multi_sim_iou_fitness(future_train, pred_future_train)
    pred_future_test = model.predict(y_test)
    future_test_fitness = wildfire.multi_sim_iou_fitness(future_test, pred_future_test)
    print(f'future_train_fitness {future_train_fitness}')
    print(f'future_test_fitness {future_test_fitness}')

    return train_fitness, test_fitness, future_train_fitness, future_test_fitness


def load_and_split_init_final_future_dataset(dataset_dir=None, func_type=None, test_size=0.5):
    # load dataset, shape = n_land, n_row, n_col, n_feat
    init_landscapes = mapsynth.load_dataset_burn_i(0, dataset_dir=dataset_dir, func_type=func_type)
    final_landscapes = mapsynth.load_dataset_burn_i(1, dataset_dir=dataset_dir, func_type=func_type)
    future_landscapes = mapsynth.load_dataset_burn_i(2, dataset_dir=dataset_dir, func_type=func_type)
    # train-test split
    x_train, x_test, y_train, y_test, future_train, future_test = train_test_split(
        init_landscapes, final_landscapes, future_landscapes, test_size=test_size)
    print('x_train.shape, x_test.shape, y_train.shape, y_test.shape')
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    return x_train, x_test, y_train, y_test, future_train, future_test


def run_grid_search_experiment(dataset_dir=None, out_dir=None, func_type='balanced_logits', n_sim=1,
                               max_time=20, seed=None, model_type='constant', pop_size=100, n_gen=100,
                               i_rep=None, n_rep=1):
    '''
    :param dataset_dir:
    :param out_dir:
    :param func_type:
    :param n_sim:
    :param max_time:
    :param seed:
    :param model_type:
    :param pop_size:
    :param n_gen:
    :param i_rep: if None, do all n_rep repetitions. Otherwise just do the single repetition specified.
    :param n_rep: the number of repetitions of the experiment.
    :return:
    '''

    # run a single rep or all the reps
    i_reps = [i_rep] if i_rep is not None else list(range(n_rep))

    for i_rep in i_reps:
        # Results
        results = {
            'dataset_dir': dataset_dir,
            'func_type': func_type, 'n_sim': n_sim, 'max_time': max_time,
            'seed': seed, 'model_type': model_type, 'out_dir': out_dir,
            'pop_size': pop_size, 'n_gen': n_gen,
        }

        # Reproducibility
        if seed is not None:
            np.random.seed(seed + i_rep)
            random.seed(seed + i_rep + 1)

        print('run_grid_search')
        print(f'model_type {model_type}')

        # load dataset
        x_train, x_test, y_train, y_test, future_train, future_test = load_and_split_init_final_future_dataset(
            dataset_dir=dataset_dir, func_type=func_type)

        if model_type == 'gp':
            model = GeneticProgrammingModel(n_sim=n_sim, max_time=max_time, pop_size=pop_size, n_gen=n_gen)
        elif model_type == 'constant':
            model = ConstantModel(n_sim=n_sim, max_time=max_time, pop_size=pop_size, n_gen=n_gen)
        elif model_type == 'logistic':
            model = LogisticModel(n_sim=n_sim, max_time=max_time, pop_size=pop_size, n_gen=n_gen)
        else:
            raise Exception('Unrecognized model_type for grid search cross-validation', model_type)

        param_grid = {'cxpb': np.linspace(0, 1, 11),
                      'mutpb': np.linspace(0, 1, 11)}
        param_grid = {'cxpb': np.linspace(0, 1, 2),
                      'mutpb': np.linspace(0, 1, 2),
                      }
        print('param_grid:', param_grid)
        gs_model = GridSearchCV(model, param_grid, cv=5, iid=False, n_jobs=-1, refit=True, error_score=np.nan)
        gs_model.fit(x_train, y_train)

        print("Grid Search Best Index:", gs_model.best_index_)
        print("Grid Search Best Score:", gs_model.best_score_)
        print('Grid Search Best Params:', gs_model.best_params_)
        print('Grid Search cross-validation results:')
        pprint(gs_model.cv_results_)

        print('Best Estimator/Model', gs_model.best_estimator_)
        if hasattr(gs_model.best_estimator_, 'hof'):
            hof = gs_model.best_estimator_.hof
            print('Best Model Hall of Fame')
            print(hof)
            results['hof'] = hof
        else:
            print('No hof')

        if hasattr(gs_model.best_estimator_, 'tree'):
            tree = gs_model.best_estimator_.tree
            print('Best Model Tree')
            print(tree)
            results['tree'] = tree
        else:
            print('No tree')

        if hasattr(gs_model.best_estimator_, 'best_ind'):
            best_ind = gs_model.best_estimator_.best_ind
            print('Best Individual')
            print(best_ind)
            results['best_ind'] = best_ind

        if hasattr(gs_model.best_estimator_, 'log'):
            log = gs_model.best_estimator_.log
            print('Best Model Log')
            print(log)
            results['log'] = log

        if hasattr(gs_model.best_estimator_, 'fitnesses'):
            fitnesses = gs_model.best_estimator_.fitnesses
            print('Best Model Fitneses')
            print(fitnesses)
            results['fitnesses'] = fitnesses

        # evaluate model
        train_fitness, test_fitness, future_train_fitness, future_test_fitness = evaluate_model_on_final_and_future(
            gs_model, x_train, x_test, y_train, y_test, future_train, future_test)

        results.update({'cxpb': gs_model.best_estimator_.cxpb, 'mutpb': gs_model.best_estimator_.mutpb,
                        'train_fitness': train_fitness, 'test_fitness': test_fitness,
                        'future_train_fitness': future_train_fitness, 'future_test_fitness': future_test_fitness,
                        'gs_model': gs_model, 'param_grid': param_grid,
                        'best_params': gs_model.best_params_, 'best_score': gs_model.best_score_,
                        'best_model': gs_model.best_estimator_})

        # Save Results
        save_repetition_results(results, out_dir, i_rep, model_type)


def run_experiment(dataset_dir=None, landscape_dir=None, func_type='balanced_logits', n_rep=None,
                   seed=None, model_type='constant', out_dir=None, i_rep=None):
    """
    Run n_rep repetitions of the experiment, using the model_type model. Or if i_rep is specified, run the ith
    repetition of the experiment. Use the func_type burn type.
    """
    # Experimental Parameters
    pop_size = 100
    n_gen = 100
    cxpb = 0.5  # crossover rate
    mutpb = 0.5  # mutation rate
    max_time = 20
    n_sim = 1

    # Reproducibility first!
    np.random.seed(seed)
    random.seed(seed + 1)

    print('run_experiment')
    print(f'model_type {model_type}')

    if (n_rep is None and i_rep is None) or (n_rep is not None and i_rep is not None):
        raise Exception(f'Exactly one of n_rep or i_rep must not be None. n_rep={n_rep}, i_rep={i_rep}.')
    elif (n_rep is not None):
        i_reps = list(range(n_rep))
    else:
        i_reps = [i_rep]

    # load dataset, shape = n_land, n_row, n_col, n_feat
    init_landscapes = mapsynth.load_dataset_burn_i(0, dataset_dir=dataset_dir, landscape_dir=landscape_dir, func_type=func_type)
    final_landscapes = mapsynth.load_dataset_burn_i(1, dataset_dir=dataset_dir, landscape_dir=landscape_dir, func_type=func_type)
    future_landscapes = mapsynth.load_dataset_burn_i(2, dataset_dir=dataset_dir, landscape_dir=landscape_dir, func_type=func_type)

    # train-test split
    test_size = 10  # 10 train, 10 test
    x_train, x_test, y_train, y_test, future_train, future_test = train_test_split(
        init_landscapes, final_landscapes, future_landscapes, test_size=test_size)
    print('x_train.shape, x_test.shape, y_train.shape, y_test.shape')
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # model
    model = get_model(model_type, n_sim=n_sim, max_time=max_time, pop_size=pop_size, n_gen=n_gen, cxpb=cxpb, mutpb=mutpb)

    for i_rep in i_reps:
        print('i_rep', i_rep)

        # train model
        model.fit(x_train, y_train)

        # evaluate model
        train_fitness, test_fitness, future_train_fitness, future_test_fitness = evaluate_model_on_final_and_future(
            model, x_train, x_test, y_train, y_test, future_train, future_test)

        results = {
            'dataset_dir': dataset_dir, 'landscape_dir': landscape_dir,
            'func_type': func_type, 'n_rep': n_rep, 'n_sim': n_sim, 'max_time': max_time,
            'seed': seed, 'model_type': model_type, 'out_dir': out_dir,
            'pop_size': pop_size, 'n_gen': n_gen, 'cxpb': cxpb, 'mutpb': mutpb,
            'train_fitness': train_fitness, 'test_fitness': test_fitness,
            'future_train_fitness': future_train_fitness, 'future_test_fitness': future_test_fitness,
            'model': model, 'best_model': model, 'i_rep': i_rep
        }
        for attr in ['fitnesses', 'hof', 'tree', 'best_ind', 'log']:
            if hasattr(model, attr):
                results[attr] = getattr(model, attr)

        # Save Results
        save_repetition_results(results, out_dir, i_rep, model_type)


def process_results():

    # plot
    plt.plot(np.arange(len(model.fitnesses)), model.fitnesses)
    plt.title('model fitness vs function evaluations')
    plt.ylabel('fitness')
    plt.xlabel('# function evaluations')
    plt.show()


def repetition_results_path(results_dir, i_rep=0, model_type=None):
    basename = f'results_model_{model_type}_rep_{i_rep}'
    return Path(results_dir) / basename


def save_repetition_results(results, out_dir=None, i_rep=0, model_type=None):
    '''
    Save results to a pickle located in out_dir, named after the repetition number.
    :param out_dir:
    :param i_rep:
    :param n_rep:
    :return:
    '''
    if out_dir is not None:
        path = repetition_results_path(out_dir, i_rep, model_type)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as fh:
            print(f'Saving results for repetition {i_rep} to {path}')
            pickle.dump(results, fh)


def load_repetition_results(results_dir, i_rep=0, model_type=None):
    path = repetition_results_path(results_dir, i_rep, model_type)
    with open(path, 'rb') as fh:
        print(f'Loading results for repetition {i_rep} from {path}')
        return pickle.load(fh)


def plot_hyperparam_search_results(results_dir, i_rep=0, model_type=None):
    results = load_repetition_results(results_dir, i_rep, model_type)
    cxpbs = results['param_grid']['cxpb']
    mutpbs = results['param_grid']['mutpb']
    n_c = len(cxpbs)
    n_m = len(mutpbs)
    c2i = {cxpbs[i]: i for i in range(n_c)}
    m2i = {mutpbs[i]: i for i in range(n_m)}
    fits = np.zeros((n_c, n_m))
    params = results['gs_model'].cv_results_['params']
    mean_test_score = results['gs_model'].cv_results_['mean_test_score']

    for i in range(len(params)):
        fits[c2i[params[i]['cxpb']], m2i[params[i]['mutpb']]] = mean_test_score[i]

    fig, ax = plt.subplots(n_c)
    for i in range(n_c):
        ax[i].plot(cxpbs, fits[i])
        ax[i].set_xlabel('mutation rate')
        ax[i].set_ylabel('fitness')
        ax[i].set_title(f'Crossover Rate={cxpbs[i]:.2}')
    plt.suptitle('Mutation Rate vs Fitness for Different Crossover Rates')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    fig, ax = plt.subplots(n_c)
    for i in range(n_m):
        ax[i].plot(cxpbs, fits[i])
        ax[i].set_xlabel('crossover rate')
        ax[i].set_ylabel('fitness')
        ax[i].set_title(f'Mutation Rate={mutpbs[i]:.2}')
    plt.suptitle('Crossover Rate vs Fitness for Different Mutation Rates')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



def plot_results_model(results_dir, i_rep=0, model_type=None):
    results = load_repetition_results(results_dir, i_rep, model_type)
    logbook = results['log']
    gen = logbook.select("gen")
    gen_max_fits = logbook.chapters["fitness"].select("max")
    gen_avg_fits = logbook.chapters['fitness'].select('avg')
    size_avgs = logbook.chapters["size"].select("avg")
    size_maxs = logbook.chapters['size'].select('max')

    fitnesses = results['fitnesses']
    fig, ax = plt.subplots()
    plt.plot(list(range(len(fitnesses))), fitnesses, label='fitness')
    plt.title('fitness vs evaluations')
    plt.legend()
    plt.xlabel('evaluation')
    plt.ylabel('fitness')
    plt.show()

    fig, ax = plt.subplots()
    plt.plot(gen, gen_max_fits, label='max')
    plt.plot(gen, gen_avg_fits, label='mean')
    plt.title('Fitness vs Generation')
    plt.legend()
    plt.xlabel('generation')
    plt.ylabel('fitness')
    plt.show()

    fig, ax = plt.subplots()
    plt.plot(gen, size_maxs, label='max')
    plt.plot(gen, size_avgs, label='mean')
    plt.title('Size vs Generation')
    plt.legend()
    plt.xlabel('generation')
    plt.ylabel('size')
    plt.show()


def animate_results_model(landscape_dir, results_dir, i_rep=0, model_type=None, seed=None, max_time=20, n_sim=1,
                          display=False, path=None):
    # reproducibility
    if seed is not None:
        np.random.seed(seed)

    results = load_repetition_results(results_dir, i_rep, model_type)
    fire_func = results['best_model'].get_fire_func()
    landscape = wildfire.make_landscape_from_dir(landscape_dir)
    landscape = mapsynth.start_fire(landscape)
    wildfire.animate_fire(landscape, fire_func, max_time=max_time, n_sim=n_sim, display=display, path=path)


if __name__ == '__main__':
    main_parser = argparse.ArgumentParser()
    subparsers = main_parser.add_subparsers()

    parser = subparsers.add_parser('run_experiment')
    parser.add_argument('--dataset-dir', help='directory containing landscape directories', default=None)
    parser.add_argument('--seed', type=int, default=0,
                        help='Used to seed the random number generator for reproducibility')
    parser.add_argument('--model-type', help='e.g constant, balanced_logits or gp)', default='constant')
    parser.add_argument('--out-dir', help='directory in which to save the results file of every repetition', default=None)
    parser.add_argument('--i-rep', type=int, default=None, help='Run the ith repetition of the experiment. ')
    parser.add_argument('--n-rep', type=int, default=None, help='Run n repetitions of the experiment')
    parser.set_defaults(func=run_experiment)

    parser = subparsers.add_parser('run_grid_search_experiment')
    parser.add_argument('--dataset-dir', help='directory containing landscape directories', default=None)
    parser.add_argument('--seed', type=int, default=0,
                        help='Used to seed the random number generator for reproducibility')
    parser.add_argument('--model-type', help='e.g constant, balanced_logits or gp)', default='constant')
    parser.add_argument('--out-dir', help='directory in which to save the results file of every repetition', default=None)
    parser.add_argument('--i-rep', type=int, default=None, help='Run the ith repetition of the experiment. ')
    parser.add_argument('--n-rep', type=int, default=1, help='Run n repetitions of the experiment')
    parser.add_argument('--pop-size', type=int, default=100, help='Population size')
    parser.add_argument('--n-gen', type=int, default=100, help='Number of generations')
    parser.set_defaults(func=run_grid_search_experiment)

    parser = subparsers.add_parser('animate_results_model')
    parser.add_argument('--landscape-dir', help='directory containing landscape layers', default=None)
    parser.add_argument('--model-type', help='e.g constant, balanced_logits or gp)', default='constant')
    parser.add_argument('--results-dir', help='directory in which to save the results file of every repetition', default=None)
    parser.add_argument('--i-rep', type=int, default=None, help='Run the ith repetition of the experiment. ')
    parser.add_argument('--max-time', type=int, help='The length of time to simulate the fire during each period')
    parser.add_argument('--n-sim', type=int, help='The number of simulations.', default=1)
    parser.add_argument('--seed', type=int, default=0,
                        help='Used to seed the random number generator for reproducibility')
    parser.add_argument('--display', default=False, action='store_true', help='Display the animation')
    parser.set_defaults(func=animate_results_model)

    parser = subparsers.add_parser('plot_results_model')
    parser.add_argument('--model-type', help='e.g constant, balanced_logits or gp)', default='constant')
    parser.add_argument('--results-dir', help='directory in which to save the results file of every repetition', default=None)
    parser.add_argument('--i-rep', type=int, default=None, help='Run the ith repetition of the experiment. ')
    parser.set_defaults(func=plot_results_model)

    parser = subparsers.add_parser('plot_hyperparam_search_results')
    parser.add_argument('--model-type', help='e.g constant, balanced_logits or gp)', default='constant')
    parser.add_argument('--results-dir', help='directory in which to save the results file of every repetition', default=None)
    parser.add_argument('--i-rep', type=int, default=None, help='Run the ith repetition of the experiment. ')
    parser.set_defaults(func=plot_hyperparam_search_results)





    args = main_parser.parse_args()
    func = args.func
    kws = vars(args)
    del kws['func']
    func(**kws)
