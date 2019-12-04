from nullexperiment import *

run_experiment(dataset_dir='data', out_dir='results/timesteps_gp_balanced',
dtype = 'timesteps', func_type='balanced_logits', max_time=10, seed = None,
model_type='gp', pop_size=100, n_gen=100, mutpb=0.1, cxpb=0.8, n_rep = 5)