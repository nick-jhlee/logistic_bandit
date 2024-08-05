"""
Created on 2024.08.04
@author: nicklee

Class for the EMK of [Emmenegger et al., NeurIPS'23]. Inherits from the LogisticBandit class.

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


def mu(z):
    return 1 / (1 + np.exp(-z))


def VAW_regularizer(arm, theta):
    """
    Computes the Vovk-Azoury-Warmuth predictor (regularizer) at theta for Bernoulli distribution
    set lambda = 1
    """
    return np.dot(theta, theta) + np.log(1 + np.exp(np.dot(arm, theta)))


def VAW_regularizer_J(arm, theta):
    """
    Computes the gradient of VAW_regularizer
    """
    return 2 * theta + mu(np.dot(arm, theta)) * arm


class EMK(LogisticBandit):
    def __init__(self, param_norm_ub, arm_norm_ub, dim, failure_level, horizon, lazy_update_fr=1, plot_confidence=False,
                 N_confidence=1000):
        """
        :param lazy_update_fr:  integer dictating the frequency at which to do the learning if we want the algo to be lazy (default: 1)
        """
        super().__init__(param_norm_ub, arm_norm_ub, dim, failure_level)
        self.name = 'EMK'
        self.lazy_update_fr = lazy_update_fr
        # initialize some learning attributes
        self.theta_hat = np.random.normal(0, 1, (self.dim, 1))
        self.ctr = 0
        self.ucb_bonus = 0
        self.log_loss_hat = 0
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

        ## Vovk-Azoury-Warmuth predictor (see their Theorem 5), implemented with SLSQP
        ineq_cons = {'type': 'ineq',
                     'fun': lambda theta: np.array([
                         self.param_norm_ub ** 2 - np.dot(theta, theta)]),
                     'jac': lambda theta: 2 * np.array([- theta])}
        opt = minimize(self.neg_log_likelihood_full, x0=np.reshape(self.theta_hat, (-1,)), method='SLSQP',
                       jac=self.neg_log_likelihood_J,
                       constraints=ineq_cons)
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
            res = np.random.normal(0, 1)
        else:
            ## SLSQP
            obj = lambda theta: -np.sum(arm * theta)
            obj_J = lambda theta: -arm
            ineq_cons = {'type': 'ineq',
                         'fun': lambda theta: np.array([
                             self.ucb_bonus - (self.neg_log_likelihood_full(theta) - self.log_loss_hat),
                             self.param_norm_ub ** 2 - np.dot(theta, theta)]),
                         'jac': lambda theta: - np.vstack((self.neg_log_likelihood_full_J(theta).T, 2 * theta))}
            opt = minimize(obj, x0=self.theta_hat, method='SLSQP', jac=obj_J, constraints=ineq_cons)
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

            ## plot confidence set
            if self.plot and len(self.rewards) == self.T - 2:
                ## store data
                interact_rng = np.linspace(-self.param_norm_ub - 0.5, self.param_norm_ub + 0.5, self.N)
                x, y = np.meshgrid(interact_rng, interact_rng)
                f = lambda x, y: self.logistic_loss_seq(np.array([x, y])) - self.log_loss_hat
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
