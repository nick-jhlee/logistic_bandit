"""
Helper functions for regret minimization
"""
import os
import numpy as np
import numpy.random as ra

from logbexp.algorithms.algo_factory import create_algo
from logbexp.bandit.logistic_env import create_env
from joblib import Parallel, delayed
from tqdm import tqdm
from time import perf_counter
import ipdb



def one_bandit_exp(config, seed:int, one_exp=False):
    env = create_env(config, seed)
    algo = create_algo(config)
    horizon = config["horizon"]
    regret_array = np.empty(horizon)
    kappa_inv_array = np.empty(horizon)
    # let's go
    t_start = perf_counter()
    for t in tqdm(range(horizon)):
        arm = algo.pull(env.arm_set)   
        reward, regret, kappa_inv = env.interact(arm) 
        regret_array[t] = regret
        kappa_inv_array[t] = kappa_inv
        algo.learn(arm, reward)
    t_stop = perf_counter()
    print(f"{algo.name:20s}: cumregret: {np.sum(regret_array):10g}, runtime: {t_stop - t_start:10.1f}")
    #print(f"Runtime of {algo.name}: ", t_stop - t_start)

    ## Plot, IF only one experiment is being run
    plotting_algos = ["OFUGLB", "OFUGLB-e", "EMK", "OFULogPlus", "OFULog-r"]
    if one_exp and config["plot_confidence"]:
        N, S = config["N_confidence"], config["param_norm_ub"]
        ## store data
        interact_rng = np.linspace(-S - 0.5, S + 0.5, N)
        X, Y = np.meshgrid(interact_rng, interact_rng)

        if algo.name not in plotting_algos:
            print(f"Plotting not implemented (or not suitable) for {algo.name}")
        else:
            if algo.name == "EMK":
                f = lambda x, y: algo.neg_log_likelihood_sequential_plotting(np.array([x, y])) - algo.weighted_log_loss_hat
                Z = (f(X, Y) <= np.log(1 / algo.failure_level)).astype(int)
            elif algo.name == "OFUGLB-e":
                tmp = np.array([X, Y]) - algo.theta_hat.reshape(2, 1, 1)
                Z = (np.einsum('kij,kl,lij->ij', tmp, algo.Ht, tmp) <= algo.ucb_bonus).astype(int)
            elif algo.name == "OFUGLB":
                f = lambda x, y: algo.neg_log_likelihood_plotting(np.array([x, y])) - algo.log_loss_hat
                Z = (f(X, Y) <= algo.ucb_bonus).astype(int)
            elif algo.name == "OFULogPlus":
                f = lambda x, y: algo.neg_log_likelihood_plotting(np.array([x, y])) - algo.log_loss_hat
                Z = (f(X, Y) <= algo.ucb_bonus).astype(int)
            elif algo.name == "OFULog-r":
                f = lambda x, y: algo.neg_log_likelihood_plotting(np.array([x, y])) - algo.log_loss_hat
                Z = (f(X, Y) <= algo.ucb_bonus).astype(int)
            if algo.name != "OFUGLB-e":
                Z_S = (np.linalg.norm(np.array([X, Y]), axis=0) <= S).astype(int)
                Z = Z_S * Z
            save_npz(X, Y, Z, algo.theta_hat, config)

    return regret_array, 1 / np.mean(kappa_inv_array, axis=0)


# {'repeat': 2,
#  'horizon': 40,
#  'dim': 2,
#  'algo_name': 'GLOC',
#  'theta_star': [7.071067811865475, 7.071067811865475],
#  'param_norm_ub': 11,
#  'failure_level': 0.05,
#  'arm_set_type': 'tv_discrete',
#  'arm_set_size': 20,
#  'arm_norm_ub': 1,
#  'norm_theta_star': 10.0,
#  'plot_confidence': False,
#  'N_confidence': 500,
#  }

def many_bandit_exps(config):
    def run_bandit_exp(idx):
        return one_bandit_exp(config, seed_ary[idx])

    seed_factory = ra.SeedSequence(config['seed'])
    seed_ary = seed_factory.generate_state(config['repeat'])

    n_jobs=6
    b_parallel = True # FIXME
    
    if config['repeat'] > 1:
        if b_parallel:
            everything = Parallel(n_jobs=n_jobs)(delayed(run_bandit_exp)(i) for i in range(config["repeat"]))
            regret = [item[0] for item in everything]
            kappa_invs = [item[1] for item in everything]
        else: 
            regret = []
            kappa_invs = []
            for i in range(config["repeat"]):
                everything = one_bandit_exp(config, seed_ary[i], True)
                regret.append(everything[0])
                kappa_invs.append(everything[1])

        cum_regret = np.cumsum(regret, axis=1)
        return np.mean(cum_regret, axis=0), np.std(cum_regret, axis=0), np.mean(kappa_invs, axis=0)
    else:
        everything = one_bandit_exp(config, seed_ary[0], True)
        regret = everything[0]
        kappa_inv = everything[1]
        cum_regret = np.cumsum(regret)
        return cum_regret, np.zeros(cum_regret.shape), kappa_inv


def save_npz(X, Y, Z, theta_hat, config):
    """
    Save the data for plotting the CS
    """
    plot_path = os.path.join('logs/plot', 'h{}d{}a{}n{}t{}.npz'.format(config["horizon"], config["dim"],
                                                    config["algo_name"],
                                                    config['norm_theta_star'],
                                                    config['arm_set_type']))

    with open(plot_path, "wb") as file:
        np.savez(file, theta_hat=theta_hat, x=X, y=Y, z=Z)
