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
from sklearn.base import BaseEstimator
import sys
import features
import wildfire
import mapsynth


class WildfireModel(BaseEstimator):
    """
    A wildfire model fire spreading function that it uses to predict/simulate landscape burns.
    """
    def __init__(self, n_sim=1, max_time=20):
        self.n_sim = n_sim
        self.max_time = max_time

    def get_fire_func(self, *args, **kwargs):
        raise NotImplementedError()

    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        return wildfire.predict(X, self.get_fire_func(), self.max_time)

    def score(self, X, y=None):
        return wildfire.evaluate(X, y, self.get_fire_func(), n_sim=self.n_sim, max_time=self.max_time)


class FixedLogitsModel(WildfireModel):
    """
    Hard-coded Logits model. No parameters. Nothing to `fit`.
    """
    def __init__(self, bias=0, feature_scale=np.array([1.0, 1.0, 1.0, 1.0]), max_time=20):
        super().__init__(max_time=max_time)
        self.bias = bias
        self.feature_scale = feature_scale

    def get_fire_func(self):
        fire_func = features.make_logits_fire_func(features.make_scaled_logits_func(self.bias, self.feature_scale))
        return fire_func

    def fit(self, X, y):
        self.fitnesses = []


class BalancedLogitsModel(FixedLogitsModel):
    def __init__(self, max_time=20):
        super().__init__(bias=-7, feature_scale=np.array([6.0, 0.2, 0.2, 0.8]), max_time=max_time)


class WindLogitsModel(FixedLogitsModel):
    def __init__(self, max_time=20):
        super().__init__(bias=-6, feature_scale=np.array([0.0, 0.0, 0.0, 3.0]), max_time=max_time)


class DzLogitsModel(FixedLogitsModel):
    def __init__(self, max_time=20):
        super().__init__(bias=-6, feature_scale=np.array([10.0, 0.0, 0.0, 0.0]), max_time=max_time)


class EvaluateWildfire:
    def __init__(self, x, y, model, n_sim=1, max_time=20, fitnesses=None):
        """
        Make a DEAP evaluation function, which takes an individual, creates a fire_func from self.get_fire_func,
        runs wildfire.evaluate using x and y, and tracks the fitness via fitnesses.
        :param x: initial landscapes
        :param y: final landscapes
        :param model: a WildfireModel object that has a `get_fire_func` method.
        :param fitnesses: a list to which the wildfire.evaluate fitness is appended.
        """
        self.x = x
        self.y = y
        self.model = model
        self.n_sim = n_sim
        self.max_time = max_time
        self.fitnesses = fitnesses if fitnesses is not None else []

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def evaluate(self, individual):
        '''
        DEAP individual fitness evaluation function.
        :param individual: a population individual
        :return: The fitness of the individual as a fitness tuple.
        '''
        fitness = wildfire.evaluate(self.x, self.y, self.model.get_fire_func(individual), self.n_sim, self.max_time)
        self.fitnesses.append(fitness)
        return fitness,  # return tuple for DEAP


class DeapModel(WildfireModel):
    """
    A DeapModel, when fit to data, uses wildfire.evaluate to evaluate individual fire spreading
    functions in the population.
    """
    def __init__(self, n_sim=1, max_time=20):
        super().__init__(max_time=max_time, n_sim=n_sim)


class ConstantModel(DeapModel):
    """
    a one-parameter model GA
    """
    def __init__(self, n_sim=1, max_time=20, pop_size=10, n_gen=10, cxpb=0.5, mutpb=0.5):
        super().__init__(n_sim=n_sim, max_time=max_time)
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.cxpb = cxpb
        self.mutpb = mutpb

    def get_fire_func(self, individual=None):
        prob = individual[0] if individual else self.best_ind[0]

        def fire_func(neighborhood):
            return prob

        return fire_func

    def fit(self, X, y):
        """
        :param X: a list or ndarray of initial landscapes
        :param y: a list or ndarray of final landscapes
        """
        print(f'fit(): pop_size={self.pop_size}, n_gen={self.n_gen}, cxpb={self.cxpb}, mutpb={self.mutpb},',
              f'n_sim={self.n_sim}, max_time={self.max_time}')

        # HACK: sklearn.model_selection.GridSearchCV calls fit multiple times in parallel.
        # creator complains when creating the same class multiple times.
        # checking to see if the class exists already, while addressing race conditions, fixes the problem.
        if not hasattr(creator, 'FitnessMax'):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))

        # HACK: sklearn.model_selection.GridSearchCV calls fit multiple times in parallel.
        # creator complains when creating the same class multiple times.
        # checking to see if the class exists already, while addressing race conditions, fixes the problem.
        if not hasattr(creator, 'Individual'):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.random)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def mutate(individual):
            individual[0] = np.random.random()
            return individual,


        # algorithm params
        toolbox.register('mate', tools.cxUniform, indpb=1.0)
        evaluate = EvaluateWildfire(x=X, y=y, model=self)
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

        pop = toolbox.population(n=self.pop_size)
        hof = tools.HallOfFame(1)
        pop, log = algorithms.eaSimple(population=pop, toolbox=toolbox, cxpb=self.cxpb, mutpb=self.mutpb,
                                       ngen=self.n_gen, stats=mstats, halloffame=hof, verbose=False)
        print(hof)

        # save best model
        self.best_ind = np.array(hof[0])
        self.log = log
        self.fitnesses = evaluate.fitnesses
        return self


class WindModel(DeapModel):
    """
    A two-parameter model that uses the wind feature
    """
    def __init__(self, n_sim=1, max_time=20, pop_size=10, n_gen=10, cxpb=0.5, mutpb=0.5):
        super().__init__(n_sim=n_sim, max_time=max_time)
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.cxpb = cxpb
        self.mutpb = mutpb

    def get_fire_func(self, individual=None):
        bias = individual[0] if individual else self.best_ind[0]
        wind_scale = individual[1] if individual else self.best_ind[1]
        feature_scale = np.array([0, 0, 0, wind_scale], dtype=float)  # use wind feature
        fire_func = features.make_logits_fire_func(features.make_scaled_logits_func(bias, feature_scale))
        return fire_func

    def fit(self, X, y):
        """
        :param X: a list or ndarray of initial landscapes
        :param y: a list or ndarray of final landscapes
        """
        print(f'fit(): pop_size={self.pop_size}, n_gen={self.n_gen}, cxpb={self.cxpb}, mutpb={self.mutpb},',
              f'n_sim={self.n_sim}, max_time={self.max_time}')

        # HACK: sklearn.model_selection.GridSearchCV calls fit multiple times in parallel.
        # creator complains when creating the same class multiple times.
        # checking to see if the class exists already, while addressing race conditions, fixes the problem.
        if not hasattr(creator, 'FitnessMax'):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))

        # HACK: sklearn.model_selection.GridSearchCV calls fit multiple times in parallel.
        # creator complains when creating the same class multiple times.
        # checking to see if the class exists already, while addressing race conditions, fixes the problem.
        if not hasattr(creator, 'Individual'):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.random)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # algorithm params
        toolbox.register('mate', tools.cxTwoPoint)
        evaluate = EvaluateWildfire(x=X, y=y, model=self)
        toolbox.register("evaluate", evaluate)
        toolbox.register("select", tools.selTournament, tournsize=2)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=2, indpb=0.5)

        # statistics
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)

        pop = toolbox.population(n=self.pop_size)
        hof = tools.HallOfFame(1)
        pop, log = algorithms.eaSimple(population=pop, toolbox=toolbox, cxpb=self.cxpb, mutpb=self.mutpb,
                                       ngen=self.n_gen, stats=mstats, halloffame=hof, verbose=False)
        print(hof)

        # save best model
        self.best_ind = np.array(hof[0])
        self.log = log
        self.fitnesses = evaluate.fitnesses



class LogisticModel(DeapModel):
    """
    a 5-parameter logistic regression model using all 4 features
    """
    def __init__(self, n_sim=1, max_time=20, pop_size=10, n_gen=10, cxpb=0.5, mutpb=0.5):
        super().__init__(n_sim=n_sim, max_time=max_time)
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.cxpb = cxpb
        self.mutpb = mutpb

    def get_fire_func(self, individual=None):
        bias = individual[0] if individual else self.best_ind[0]
        # use dz, temp, hum, wind features
        feature_scale = np.array(individual[1:] if individual else self.best_ind[1:], dtype=float)
        fire_func = features.make_logits_fire_func(features.make_scaled_logits_func(bias, feature_scale))
        return fire_func

    def fit(self, X, y):
        """
        :param x: a list or ndarray of initial landscapes
        :param y: a list or ndarray of final landscapes
        """
        print(f'fit(): pop_size={self.pop_size}, n_gen={self.n_gen}, cxpb={self.cxpb}, mutpb={self.mutpb},',
              f'n_sim={self.n_sim}, max_time={self.max_time}')

        n_gene = 5 # one gene for each of: bias and 4 feature coefficients
        indpb = 1 / n_gene  # expected number of mutated loci per genome = 1
        tournsize = 2
        mut_mu = 0
        mut_sigma = 1

        # HACK: sklearn.model_selection.GridSearchCV calls fit multiple times in parallel.
        # creator complains when creating the same class multiple times.
        # checking to see if the class exists already, while addressing race conditions, fixes the problem.
        if not hasattr(creator, 'FitnessMax'):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))

        # HACK: sklearn.model_selection.GridSearchCV calls fit multiple times in parallel.
        # creator complains when creating the same class multiple times.
        # checking to see if the class exists already, while addressing race conditions, fixes the problem.
        if not hasattr(creator, 'Individual'):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.random)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_gene)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # algorithm params
        toolbox.register('mate', tools.cxTwoPoint)
        evaluate = EvaluateWildfire(x=X, y=y, model=self)
        toolbox.register("evaluate", evaluate)
        toolbox.register("select", tools.selTournament, tournsize=tournsize)
        toolbox.register("mutate", tools.mutGaussian, mu=mut_mu, sigma=mut_sigma, indpb=indpb)

        # statistics
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)

        pop = toolbox.population(n=self.pop_size)
        hof = tools.HallOfFame(1)
        pop, log = algorithms.eaSimple(population=pop, toolbox=toolbox, cxpb=self.cxpb, mutpb=self.mutpb,
                                       ngen=self.n_gen, stats=mstats, halloffame=hof, verbose=False)
        print(hof)

        # save best model
        self.best_ind = np.array(hof[0])
        self.log = log
        self.fitnesses = evaluate.fitnesses


def protectedDiv(left, right):
    if right != 0:
        return left / right
    else:
        return 1


# HACK: sklearn.model_selection.GridSearchCV calls fit multiple times in parallel.
# pset.addEphemeralConstant freaks out if called multiple times with a lambda function,
# or a dynamically constructed function or an instance method. It does not freak out when
# called multiple times with a module level function.
# TODO: How can we do hyperparameter search for eph_const_min and eph_const_max?
def ephemeral_uniform(eph_const_min=-10, eph_const_max=10):
    return random.random() * (eph_const_max - eph_const_min) + eph_const_min


# HACK: sklearn.model_selection.GridSearchCV calls fit multiple times in parallel.
# pset.addEphemeralConstant freaks out if called multiple times with a lambda function,
# or a dynamically constructed function or an instance method. It does not freak out when
# called multiple times with a module level function.
# TODO: How can we do hyperparameter search for eph_const_sigma?
def ephemeral_normal(eph_const_sigma=1):
    return np.random.randn() * eph_const_sigma


def compile(expr, arguments, context):
    """
    This code was copied from DEAP gp.py and modified slightly to be less DEAP specific.

    Compile the expression *expr*.

    :param expr: Expression to compile. It can either be a PrimitiveTree,
                 a string of Python code or any object that when
                 converted into string produced a valid Python code
                 expression.
    :param arguments: a list of argument names of expr.
    :param context: the mapping of the function names in expr to functions (like locals())
    :returns: a function if the primitive set has 1 or more arguments,
              or return the results produced by evaluating the tree.
    """
    code = str(expr)
    if len(arguments) > 0:
        # This section is a stripped version of the lambdify
        # function of SymPy 0.6.6.
        args = ",".join(arg for arg in arguments)
        code = "lambda {args}: {code}".format(args=args, code=code)
    try:
        return eval(code, context, {})
    except MemoryError:
        _, _, traceback = sys.exc_info()
        raise MemoryError("DEAP : Error in tree evaluation :"
                            " Python cannot evaluate a tree higher than 90. "
                            "To avoid this problem, you should use bloat control on your "
                            "operators. See the DEAP documentation for more information. "
                            "DEAP will now abort.").with_traceback(traceback)


class GeneticProgrammingModel(DeapModel):
    def __init__(self, n_sim=1, max_time=20, pop_size=10, n_gen=10, cxpb=0.5, mutpb=0.5):
        super().__init__(n_sim=n_sim, max_time=max_time)
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.cxpb = cxpb
        self.mutpb = mutpb

    def get_fire_func(self, individual=None):
        """
        Return a fire spreading function, which takes a neighborhood as input and returns the probability of
        transitioning to the fire state.
        :param individual: a population member
        :return: a function
        """
        if individual:
            func = compile(individual, arguments=self.compile_arguments, context=self.compile_context)
        else:
            func = compile(**self.best_ind)

        def feature_func(features):
            """
            :param features: ndarray of 4 features, from `features.make_logits_fire_func`.
            """
            return func(*features)

        fire_func = features.make_logits_fire_func(feature_func)
        return fire_func

    def fit(self, X, y):
        print(f'fit(): pop_size={self.pop_size}, n_gen={self.n_gen}, cxpb={self.cxpb}, mutpb={self.mutpb},',
              f'n_sim={self.n_sim}, max_time={self.max_time}')

        # The primitives of the tree and their arities.
        pset = gp.PrimitiveSet("main", 4, prefix='x')
        pset.renameArguments(x0='dz', x1='temp', x2='hum', x3='wind')
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(protectedDiv, 2)
        pset.addPrimitive(operator.neg, 1)
        pset.addPrimitive(math.cos, 1)
        pset.addPrimitive(math.sin, 1)
        # Issue: log fails when passed negative numbers.
        # pset.addPrimitive(math.log, 1)
        # Ephemeral constants randomly generate constants to be inserted into trees
        pset.addEphemeralConstant("randunif", ephemeral_uniform)
        pset.addEphemeralConstant("randnorm", ephemeral_normal)

        # HACK: sklearn.model_selection.GridSearchCV calls fit multiple times in parallel.
        # creator complains when creating the same class multiple times.
        # checking to see if the class exists already, while addressing race conditions, fixes the problem.
        if not hasattr(creator, 'FitnessMax'):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))

        # HACK: sklearn.model_selection.GridSearchCV calls fit multiple times in parallel.
        # creator complains when creating the same class multiple times.
        # checking to see if the class exists already, while addressing race conditions, fixes the problem.
        if not hasattr(creator, 'Individual'):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=pset)

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=3)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        self.compile_context = dict(**pset.context)  # convert to normal objects for pickling
        self.compile_arguments = list(pset.arguments)  # convert to normal objects for pickling
        toolbox.register("compile", compile, arguments=self.compile_arguments, context=self.compile_context)

        evaluate = EvaluateWildfire(x=X, y=y, model=self)
        toolbox.register("evaluate", evaluate)
        toolbox.register("select", tools.selTournament, tournsize=2)
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

        pop = toolbox.population(n=self.pop_size)
        hof = tools.HallOfFame(1)
        pop, log = algorithms.eaSimple(population=pop, toolbox=toolbox, cxpb=self.cxpb, mutpb=self.mutpb,
                                       ngen=self.n_gen, stats=mstats, halloffame=hof, verbose=False)
        tree = gp.PrimitiveTree(hof[0])
        print('tree', tree)
        print('hof', hof)
        print('log', log)
        print('fitnesses', evaluate.fitnesses)

        # save best model
        self.best_ind = {'expr': str(tree), 'context': self.compile_context, 'arguments': self.compile_arguments}
        # self.hof = hof
        # self.best_ind = hof[0]
        # self.best_tree = gp.PrimitiveTree(self.best_ind)
        self.log = log
        self.fitnesses = evaluate.fitnesses


