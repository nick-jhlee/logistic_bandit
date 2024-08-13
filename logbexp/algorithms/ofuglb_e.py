"""
Created on 2024.08.03
@author: nicklee

Class for the OFUGLB-e (ellipsoidal OFUGLB) of [Lee et al., arXiv'24]. Inherits from the LogisticBandit class.
Use closed-form bonus! (Theorem 3.2)

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

import math

import numpy as np
from scipy.optimize import minimize
# from cvxpygen import cpg
import warnings

from logbexp.algorithms.logistic_bandit_algo import LogisticBandit


def mu(z):
    return 1 / (1 + np.exp(-z))


def dmu(z):
    return mu(z) * (1 - mu(z))


class OFUGLBe(LogisticBandit):
    def __init__(self, param_norm_ub, arm_norm_ub, dim, failure_level, horizon, lazy_update_fr=1):
        """
        :param lazy_update_fr:  integer dictating the frequency at which to do the learning if we want the algo to be lazy (default: 1)
        """
        super().__init__(param_norm_ub, arm_norm_ub, dim, failure_level)
        self.name = 'OFUGLB-e'
        self.lazy_update_fr = lazy_update_fr
        # initialize some learning attributes
        self.theta_hat = np.random.normal(0, 1, (self.dim, 1))
        self.l2_reg = (1 + param_norm_ub) / (2 * param_norm_ub ** 2)
        self.Ht = self.l2_reg * np.eye(self.dim)

        self.ctr = 0
        self.ucb_bonus = 0
        self.log_loss_hat = 0
        # containers
        self.arms = []
        self.rewards = []
        self.T = horizon

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
        self.arms.append(arm)
        self.rewards.append(reward)

        ## SLSQP
        ineq_cons = {'type': 'ineq',
                     'fun': lambda theta: np.array([
                         self.param_norm_ub ** 2 - np.dot(theta, theta)]),
                     'jac': lambda theta: 2 * np.array([- theta])}
        opt = minimize(self.neg_log_likelihood_full, x0=np.reshape(self.theta_hat, (-1,)), method='SLSQP',
                       jac=self.neg_log_likelihood_full_J,
                       constraints=ineq_cons)
        self.theta_hat = opt.x
        # update regularized Ht
        self.Ht = self.l2_reg * np.eye(self.dim)
        for arm in self.arms:
            self.Ht += dmu(np.dot(arm, self.theta_hat)) * np.outer(arm, arm)
        # update counter
        self.ctr += 1

    def pull(self, arm_set):
        self.update_ucb_bonus()
        self.log_loss_hat = self.neg_log_likelihood_full(self.theta_hat)
        arm = np.reshape(arm_set.argmax(self.compute_optimistic_reward), (-1,))
        return arm

    # see Theorem 3.2
    def update_ucb_bonus(self):
        Lt = (1 + self.param_norm_ub / 2) * (len(self.rewards) - 1)
        beta_t2 = np.log(1 / self.failure_level) + self.dim * np.log(
            max(math.e, 2 * math.e * self.param_norm_ub * Lt / self.dim))
        self.ucb_bonus = 2 * (1 + self.param_norm_ub) * (1 + beta_t2)

    def compute_optimistic_reward(self, arm):
        if self.ctr <= 1:
            res = np.random.normal(0, 1)
        else:
            res = np.dot(self.theta_hat, arm) + np.sqrt(self.ucb_bonus * np.dot(np.dot(arm, np.linalg.inv(self.Ht)), arm))
        return res
