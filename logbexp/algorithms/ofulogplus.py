"""
Created on 10/22/23
@author: nicklee

Class for the OFULog+ of [Lee et al., AISTATS'24]. Inherits from the LogisticBandit class.

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
from scipy.optimize import minimize

from logbexp.algorithms.logistic_bandit_algo import LogisticBandit


def mu(z):
    return 1 / (1 + np.exp(-z))


class OFULogPlus(LogisticBandit):
    def __init__(self, param_norm_ub, arm_norm_ub, dim, failure_level, horizon, lazy_update_fr=1, tol=1e-7):
        """
        :param lazy_update_fr:  integer dictating the frequency at which to do the learning if we want the algo to be lazy (default: 1)
        """
        super().__init__(param_norm_ub, arm_norm_ub, dim, failure_level)
        self.name = 'OFULogPlus'
        self.lazy_update_fr = lazy_update_fr
        # initialize some learning attributes
        self.theta_hat = np.zeros((self.dim,))
        self.ctr = 0
        self.ucb_bonus = 0
        self.log_loss_hat = 0
        self.T = horizon
        self.tol = tol

    def reset(self):
        """
        Resets the underlying learning algorithm
        """
        self.theta_hat = np.zeros((self.dim,))
        self.ctr = 1
        self.arms = np.zeros((0, self.dim))
        self.rewards = np.zeros((0,))

    def learn(self, arm, reward):
        """
        Updates estimator.
        """
        self.arms = np.vstack((self.arms, arm))
        self.rewards = np.concatenate((self.rewards, [reward]))

        ## SLSQP
        ineq_cons = {'type': 'ineq',
                     'fun': lambda theta: np.array([
                         self.param_norm_ub ** 2 - np.dot(theta, theta)]),
                     'jac': lambda theta: 2 * np.array([- theta])}
        opt = minimize(self.neg_log_likelihood_full, x0=np.reshape(self.theta_hat, (-1,)), method='SLSQP',
                       jac=self.neg_log_likelihood_full_J,
                       constraints=ineq_cons, tol=self.tol)
        self.theta_hat = opt.x

        # update counter
        self.ctr += 1

    def pull(self, arm_set):
        self.update_ucb_bonus()
        self.log_loss_hat = self.neg_log_likelihood_full(self.theta_hat)
        arm = np.reshape(arm_set.argmax(self.compute_optimistic_reward), (-1,))
        return arm

    def update_ucb_bonus(self):
        self.ucb_bonus = 10 * self.dim * np.log(
            np.e + (self.param_norm_ub * len(self.rewards) / (2 * self.dim))) + 2 * (
                                 np.e - 2 + self.param_norm_ub) * np.log(1 / self.failure_level)

    def compute_optimistic_reward(self, arm):
        if self.ctr == 1:
            res = np.linalg.norm(arm)
        else:
            ## SLSQP
            obj = lambda theta: -np.sum(arm * theta)
            obj_J = lambda theta: -arm
            ineq_cons = {'type': 'ineq',
                         'fun': lambda theta: np.array([
                             self.ucb_bonus - (self.neg_log_likelihood_full(theta) - self.log_loss_hat),
                             self.param_norm_ub ** 2 - np.dot(theta, theta)]),
                         'jac': lambda theta: - np.vstack((self.neg_log_likelihood_full_J(theta).T, 2 * theta))}
            opt = minimize(obj, x0=self.theta_hat, method='SLSQP', jac=obj_J, constraints=ineq_cons, tol=self.tol)
            res = np.sum(arm * opt.x)
        return res
