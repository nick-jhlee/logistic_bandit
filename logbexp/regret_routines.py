
"""
Helper functions for regret minimization
"""

import numpy as np

from logbexp.algorithms.algo_factory import create_algo
from logbexp.bandit.logistic_env import create_env
from joblib import Parallel, delayed
from tqdm import tqdm



def one_bandit_exp(config):
    env = create_env(config)
    algo = create_algo(config)
    horizon = config["horizon"]
    regret_array = np.empty(horizon)
    kappa_inv_array = np.empty(horizon)
    # let's go
    for t in tqdm(range(horizon)):
        arm = algo.pull(env.arm_set)
        reward, regret, kappa_inv = env.interact(arm)
        regret_array[t] = regret
        kappa_inv_array[t] = kappa_inv
        algo.learn(arm, reward)
    return (regret_array, 1 / np.mean(kappa_inv_array, axis=0))


def many_bandit_exps(config):
    def run_bandit_exp(*args):
        return one_bandit_exp(config)
    # n_jobs=10
    if config['repeat'] > 1:
        everything = Parallel(n_jobs=-1)(delayed(run_bandit_exp)(i) for i in range(config["repeat"]))
        regret = [item[0] for item in everything]
        kappa_invs = [item[1] for item in everything]
        cum_regret = np.cumsum(regret, axis=1)
        return np.mean(cum_regret, axis=0), np.mean(kappa_invs, axis=0)
    else:
        everything = one_bandit_exp(config)
        regret = everything[0]
        kappa_inv = everything[1]
        cum_regret = np.cumsum(regret)
        return cum_regret, kappa_inv
    
