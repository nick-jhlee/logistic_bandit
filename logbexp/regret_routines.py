"""
Helper functions for regret minimization
"""
import os
import numpy as np

from logbexp.algorithms.algo_factory import create_algo
from logbexp.bandit.logistic_env import create_env
from joblib import Parallel, delayed
from tqdm import tqdm
from time import perf_counter


def one_bandit_exp(config, one_exp=False):
    env = create_env(config)
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
    print(f"Runtime of {algo.name}: ", t_stop - t_start)

    ## Plot, IF only one experiment is being run
    if one_exp and config["plot_confidence"]:
        N, S = config["N_confidence"], config["param_norm_ub"]
        ## store data
        interact_rng = np.linspace(-S - 0.5, S + 0.5, N)
        X, Y = np.meshgrid(interact_rng, interact_rng)
        Z_S = (np.linalg.norm(np.array([X, Y]), axis=0) <= S).astype(int)
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
            f = lambda x, y: algo.logistic_loss_seq(np.array([x, y])) - algo.log_loss_hat
            Z = (f(X, Y) <= algo.ucb_bonus).astype(int)
        else:
            raise NotImplementedError(f"Plotting not implemented for {algo.name}")
        if algo.name != "OFUGLB-e":
            Z = Z_S * Z
        save_npz(X, Y, Z, algo.theta_hat, S, config["arm_set_type"], algo.name)

    return regret_array, 1 / np.mean(kappa_inv_array, axis=0)


def many_bandit_exps(config):
    def run_bandit_exp(*args):
        return one_bandit_exp(config)

    # n_jobs=10
    if config['repeat'] > 1:
        everything = Parallel(n_jobs=-1)(delayed(run_bandit_exp)(i) for i in range(config["repeat"]))
        regret = [item[0] for item in everything]
        kappa_invs = [item[1] for item in everything]
        cum_regret = np.cumsum(regret, axis=1)
        return np.mean(cum_regret, axis=0), np.std(cum_regret, axis=0), np.mean(kappa_invs, axis=0)
    else:
        everything = one_bandit_exp(config, True)
        regret = everything[0]
        kappa_inv = everything[1]
        cum_regret = np.cumsum(regret)
        return cum_regret, np.zeros(cum_regret.shape), kappa_inv


def save_npz(X, Y, Z, theta_hat, S, ast, name):
    """
    Save the data for plotting the CS
    """
    path = f"S={S}/{ast}"
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f"{path}/{name}.npz", "wb") as file:
        np.savez(file, theta_hat=theta_hat, x=X, y=Y, z=Z)
    print(f"Saved data for {name}, S={S}, {ast}")
