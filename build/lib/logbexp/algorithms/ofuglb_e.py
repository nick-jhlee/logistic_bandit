"""
Created on 2024.08.03
@author: nicklee

Class for the OFUGLB-e (ellipsoidal OFUGLB) of [Lee et al., arXiv'24]. Inherits from the LogisticBandit class.

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
import cvxpy as cp
from cvxpygen import cpg
import warnings

from logbexp.algorithms.logistic_bandit_algo import LogisticBandit


def mu(z):
    return 1 / (1 + np.exp(-z))


def dmu(z):
    return mu(z) * (1 - mu(z))


class OFUGLBe(LogisticBandit):
    def __init__(self, param_norm_ub, arm_norm_ub, dim, failure_level, horizon, lazy_update_fr=1, plot_confidence=False,
                 N_confidence=1000):
        """
        :param lazy_update_fr:  integer dictating the frequency at which to do the learning if we want the algo to be lazy (default: 1)
        """
        super().__init__(param_norm_ub, arm_norm_ub, dim, failure_level)
        self.name = 'OFUGLB-e'
        self.lazy_update_fr = lazy_update_fr
        # initialize some learning attributes
        self.theta_hat = np.random.normal(0, 1, (self.dim, 1))
        self.Ht = ((1 + param_norm_ub) / (2 * param_norm_ub ** 2)) * np.eye(self.dim)
        self.ctr = 0
        self.ucb_bonus = 0
        self.log_loss_hat = 0
        # containers
        self.arms = []
        self.rewards = []
        self.plot = plot_confidence
        self.N = N_confidence
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
        self.Ht = ((1 + self.param_norm_ub) / (2 * self.param_norm_ub ** 2)) * np.eye(self.dim)
        for s, arm in enumerate(self.arms):
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
            ## CVXPY (splitting conic solver, SCS)
            arm_res = np.reshape(arm, (-1, 1))
            theta_cvxpy = cp.Variable((self.dim, 1))
            theta_cvxpy.value = np.reshape(self.theta_hat, (-1, 1))
            constraints = [cp.norm(theta_cvxpy, 2) <= self.param_norm_ub,
                           # cp.sum_squares(theta_cvxpy) <= self.param_norm_ub ** 2,
                           cp.quad_form(theta_cvxpy - np.reshape(self.theta_hat, (-1, 1)), self.Ht) <= self.ucb_bonus]
            problem = cp.Problem(cp.Maximize(cp.sum(cp.multiply(theta_cvxpy, arm_res))), constraints)
            # cpg.generate_code(problem, code_dir='MLE', solver='SCS')
            # from MLE.cpg_solver import cpg_solve
            # problem.register_solve('CPG', cpg_solve)
            # problem.solve(method='CPG', warm_start=True)
            problem.solve(solver=cp.SCS, warm_start=True)
            res = problem.value
            if problem.status != 'optimal':
                warnings.warn(f"CVXPY failed to converge with status {problem.status}.. using SLSQP")
                ## SLSQP
                obj = lambda theta: -np.sum(arm * theta)
                obj_J = lambda theta: -arm
                ineq_cons = {'type': 'ineq',
                             'fun': lambda theta: np.array([
                                 self.ucb_bonus - (theta - self.theta_hat).T @ self.Ht @ (theta - self.theta_hat),
                                 self.param_norm_ub ** 2 - np.dot(theta, theta)]),
                             'jac': lambda theta: 2 * np.array([self.Ht @ (theta - self.theta_hat), - theta])}
                opt = minimize(obj, x0=self.theta_hat, method='SLSQP', jac=obj_J, constraints=ineq_cons)
                res = np.sum(arm * opt.x)

            ## Trust-Regions Constrained Optim (scipy.optimize)
            # obj = lambda theta: -np.sum(arm * theta)
            # obj_J = lambda theta: -arm
            # obj_H = lambda theta: np.zeros((self.dim, self.dim))
            # cons_f = lambda theta: [np.dot(theta, theta), (theta - self.theta_hat).T @ self.Ht @ (theta - self.theta_hat)]
            # cons_J = lambda theta: [2 * theta, 2 * (theta - self.theta_hat) @ self.Ht]
            # cons_H = lambda theta, v: 2*(v[0] * np.eye(self.dim) + v[1] * self.Ht)
            # constraints_scipy = NonlinearConstraint(cons_f, -np.inf, [self.param_norm_ub ** 2, self.ucb_bonus], jac=cons_J, hess=cons_H)
            # opt = minimize(obj, x0=self.theta_hat, method='trust-constr', jac=obj_J, hess=obj_H, constraints=constraints_scipy)
            # res = np.sum(arm * opt.x)

            ## plot confidence set
            if self.plot and len(self.rewards) == self.T - 2:
                ## store data
                interact_rng = np.linspace(-self.param_norm_ub - 0.5, self.param_norm_ub + 0.5, self.N)
                x, y = np.meshgrid(interact_rng, interact_rng)
                f = lambda x, y: (np.array([x, y]) - self.theta_hat).T @ self.Ht @ (np.array([x, y]) - self.theta_hat)
                z = (f(x, y) <= self.ucb_bonus) & (np.linalg.norm(np.array([x, y]), axis=0) <= self.param_norm_ub)
                z = z.astype(int)
                with open(f"S={self.param_norm_ub}/{self.name}.npz", "wb") as file:
                    np.savez(file, theta_hat=self.theta_hat, x=x, y=y, z=z)
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
