"""
Created on 2024.08.03
@author: nirjhar-das, nicklee
https://github.com/nirjhar-das/GLBandit_Limited_Adaptivity

Class for the RS-GLinCB of [Sawarni et al., arXiv'24]. Inherits from the LogisticBandit class.

Additional Attributes
---------------------
lazy_update_fr : int
    integer dictating the frequency at which to do the learning if we want the algo to be lazy
theta_hat : np.array(dim)
    maximum-likelihood estimator
log_loss_hat : float
    log-loss at current estimate theta_hat
ctr : int
    counter for lazy updates
"""
import numpy as np
import numpy.linalg as LA
from scipy.optimize import minimize, NonlinearConstraint

import math

from logbexp.algorithms.logistic_bandit_algo import LogisticBandit


def logistic(z):
    return np.log(1 + np.exp(z))


def mu(z):
    return 1 / (1 + np.exp(-z))


def dmu(z):
    return mu(z) * (1 - mu(z))


class RS_GLinCB(LogisticBandit):
    def __init__(self, param_norm_ub, arm_norm_ub, dim, failure_level, horizon, lazy_update_fr=1, plot_confidence=False,
                 N_confidence=1000):
        """
        :param lazy_update_fr:  integer dictating the frequency at which to do the learning if we want the algo to be lazy (default: 1)
        """
        super().__init__(param_norm_ub, arm_norm_ub, dim, failure_level)
        self.name = 'RS-GLinCB'
        self.lazy_update_fr = lazy_update_fr
        # initialize some learning attributes
        self.ctr = 1
        self.ucb_bonus = 0
        self.log_loss_hat = 0
        # containers
        self.plot = plot_confidence
        self.N = N_confidence
        self.T = horizon

        self.triggered_arms = []
        self.triggered_rewards = []
        self.nontriggered_arms = []
        self.nontriggered_rewards = []
        self.tau = 1
        self.l2reg = dim * np.log(horizon / failure_level)
        self.gamma = 25 * param_norm_ub * np.sqrt(dim * np.log(horizon / failure_level))
        self.V = self.l2reg * np.eye(dim)
        self.V_inv = (1 / self.l2reg) * np.eye(dim)
        self.H = self.l2reg * np.eye(dim)
        self.H_inv = (1 / self.l2reg) * np.eye(dim)
        self.H_tau = self.l2reg * np.eye(dim)
        self.H_tau_inv = (1 / self.l2reg) * np.eye(dim)
        self.theta_hat_o = np.random.normal(0, 1, (self.dim, 1))
        self.theta_hat_tau = np.random.normal(0, 1, (self.dim, 1))
        self.theta_tilde = np.random.normal(0, 1, (self.dim, 1))
        self.switch1, self.switch2 = False, False

        # compute an upper bound on kappa
        self.kappa = 3 + np.exp(param_norm_ub)
        self.warmup_threshold = 1 / (self.gamma ** 2 * self.kappa)

    def reset(self):
        """
        Resets the underlying learning algorithm
        """
        self.theta_hat = np.random.normal(0, 1, (self.dim, 1))
        self.ctr = 1
        self.arms = []
        self.rewards = []

    def learn(self, arm, reward):
        """
        Updates estimator.
        """
        if self.switch1:
            self.triggered_arms.append(arm)
            self.triggered_rewards.append(reward)
            self.V += np.outer(arm, arm)
            # Sherman-Morrison formula
            self.V_inv -= np.outer(self.V_inv @ arm, self.V_inv @ arm) / (1 + arm.T @ self.V_inv @ arm)
            # compute theta_hat_o
            obj = lambda theta: (self.neg_log_likelihood(theta, self.triggered_arms, self.triggered_rewards)
                                 + (self.l2reg / 2) * np.dot(theta, theta))
            opt = minimize(obj, x0=np.reshape(self.theta_hat_o, (-1,)))
            self.theta_hat_o = opt.x
            self.switch1 = False
        else:
            self.nontriggered_arms.append(arm)
            self.nontriggered_rewards.append(reward)

    def pull(self, arm_set):
        # check if warmup is satisfied
        f_tmp = lambda x: x.T @ self.V_inv @ x
        if max(f_tmp(arm_set)) >= self.warmup_threshold:  # (Switching Criterion I)
            arm = arm_set.argmax(f_tmp)
            self.switch1 = True
            return np.reshape(arm, (-1,))
        else:
            if LA.det(self.H) > 2 * LA.det(self.H_tau):  # (Switching Criterion II)
                self.H_tau = self.H
                self.H_tau_inv = self.H_inv
                # compute theta_hat_tau via norm-constrained MLE
                obj = lambda theta: self.neg_log_likelihood(theta, self.nontriggered_arms, self.nontriggered_rewards)
                cstrf_norm = lambda theta: np.linalg.norm(theta)
                constraint_norm = NonlinearConstraint(cstrf_norm, 0, self.param_norm_ub)
                opt = minimize(obj, x0=np.reshape(self.theta_hat, (-1,)), method='SLSQP', constraints=[constraint_norm])
                self.theta_hat_tau = opt.x
            # arm elimination
            UCB_o = lambda x: np.dot(x, self.theta_hat_o) + self.gamma * np.sqrt(self.kappa) * np.sqrt(x.T @ self.V_inv @ x)
            LCB_o = lambda x: np.dot(x, self.theta_hat_o) - self.gamma * np.sqrt(self.kappa) * np.sqrt(x.T @ self.V_inv @ x)
            lcb = min(LCB_o(arm_set))
            arm_set = arm_set[UCB_o(arm_set) > lcb]
            # UCB
            UCB = lambda x: np.dot(x, self.theta_hat_tau) + 150 * np.sqrt(x.T @ self.H_tau_inv @ x) * np.sqrt(
                self.dim * np.log(self.T / self.failure_level))
            arm = arm_set.argmax(UCB)
            self.H += (dmu(np.dot(arm, self.theta_hat_o)) / math.e) * np.outer(arm, arm)
            # Sherman-Morrison formula
            tmp = np.sqrt(dmu(np.dot(arm, self.theta_hat_o)) / math.e) * arm
            self.H_inv -= np.outer(self.V_inv @ tmp, self.V_inv @ tmp) / (1 + tmp.T @ self.V_inv @ tmp)

    def neg_log_likelihood(self, theta, arms, rewards):
        """
        Computes the log-loss for triggered timesteps
        """
        if len(self.triggered_rewards) == 0:
            return 0
        else:
            X = np.array(arms)
            r = np.array(rewards).reshape((-1, 1))
            theta = theta.reshape((-1, 1))
            # print(X.shape, r.shape)
            return np.sum(r * logistic(-X @ theta) + (1 - r) * logistic(X @ theta))
