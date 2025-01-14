"""
Created on 2024.05.21
@author: nicklee

Class for the OFUGLB of [Lee et al., NeurIPS'24]. Inherits from the LogisticBandit class.

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

from logbexp.algorithms.logistic_bandit_algo import LogisticBandit

#import line_profiler
import warnings
warnings.filterwarnings("error", category=RuntimeWarning)
import ipdb

class OFUGLB(LogisticBandit):
    def __init__(self, param_norm_ub, arm_norm_ub, dim, failure_level, horizon, lazy_update_fr=1, tol=1e-7):
        """
        :param lazy_update_fr:  integer dictating the frequency at which to do the learning if we want the algo to be lazy (default: 1)
        """
        super().__init__(param_norm_ub, arm_norm_ub, dim, failure_level)
        self.name = 'OFUGLB'
        self.lazy_update_fr = lazy_update_fr
        # initialize some learning attributes
        self.theta_hat = np.zeros(self.dim)
        self.ctr = 0
        self.ucb_bonus = 0
        self.log_loss_hat = 0
        self.T = horizon
        self.tol = tol

    def reset(self):
        """
        Resets the underlying learning algorithm
        """
        self.theta_hat = np.zeros(self.dim)
        self.ctr = 1
        self.arms = np.zeros((0, self.dim))
        self.rewards = np.zeros((0,))

    def learn(self, arm, reward):
        """
        Updates estimator.
        """
        self.arms = np.vstack((self.arms, arm))
        self.rewards = np.concatenate((self.rewards, [reward]))

        # SLSQP
        ineq_cons = {   'type': 'ineq',
                        'fun': lambda theta: 
                               self.param_norm_ub ** 2 - np.dot(theta, theta),
                        'jac': lambda theta: 2 * theta
                    }
        opt = minimize(self.neg_log_likelihood_full, x0=self.theta_hat, method='SLSQP',
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

    # see Theorem 3.1
    def update_ucb_bonus(self):
        Lt = (1 + self.param_norm_ub / 2) * (len(self.rewards) - 1)
        self.ucb_bonus = np.log(1 / self.failure_level) + self.dim * np.log(
            max(math.e, 2 * math.e * self.param_norm_ub * Lt / self.dim))

    def compute_optimistic_reward(self, arm):
        if self.ctr <= 1:
            res = np.linalg.norm(arm)   # at t = 1, pull the arm with the largest norm
        else:
            def ff(theta):
                f1 = self.ucb_bonus - (self.neg_log_likelihood_full(theta) - self.log_loss_hat)
                f2 = self.param_norm_ub ** 2 - np.dot(theta, theta)
                return np.array([f1, f2])

            def gg(theta):
                return -np.vstack((self.neg_log_likelihood_full_J(theta).T, 2 * theta))

            ## SLSQP
            obj = lambda theta: -np.sum(arm * theta)
            obj_J = lambda theta: -arm
            ineq_cons = {'type': 'ineq',
                         'fun': ff,
                         'jac': gg, }
            opt = minimize(obj, x0=self.theta_hat, method='SLSQP', jac=obj_J, constraints=ineq_cons, tol=self.tol)
            res = np.sum(arm * opt.x)

            ## CVXPY
            # arm_res = np.reshape(arm, (-1, 1))
            # theta = cp.Variable((self.dim, 1))
            # constraints = [cp.norm(theta, 2) <= self.param_norm_ub,
            #                self.neg_log_likelihood(theta) - self.neg_log_likelihood(self.theta_hat) <= self.ucb_bonus]
            # objective = cp.Maximize(cp.sum(cp.multiply(theta, arm_res)))
            # problem = cp.Problem(objective, constraints)
            # try:
            #     problem.solve(solver=cp.SCS)
            # except:
            #     try:
            #         problem.solve(solver=cp.ECOS, reltol=1e-4)
            #     except:
            #         # print("Optimization failed")
            #         print(len(self.rewards))
            #         print(problem.status)
            #         problem.solve(tol_gap_rel=1e-4, verbose=True)
            #         # raise ValueError
            # res = problem.value
        return res

    # def neg_log_likelihood_cp(self, theta):
    #     """
    #     Computes the full log-loss estimated at theta
    #     CVXPY version
    #     """
    #     if len(self.rewards) == 0:
    #         return 0
    #     else:
    #         X = np.array(self.arms)
    #         r = np.array(self.rewards).reshape((-1, 1))
    #         theta = theta.reshape((-1, 1))
    #         # print(X.shape, r.shape)
    #         return cp.sum(cp.multiply(r, cp.logistic(-X @ theta)) + cp.multiply((1 - r), cp.logistic(X @ theta)))
