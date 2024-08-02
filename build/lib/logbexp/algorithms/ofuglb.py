"""
Created on 2024.05.21
@author: nicklee

Class for the OFUGLB of [Lee et al., arXiv'24]. Inherits from the LogisticBandit class.

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
import cvxpy as cp
from scipy.optimize import minimize, NonlinearConstraint

import math

from logbexp.algorithms.logistic_bandit_algo import LogisticBandit


def logistic(z):
    return np.log(1 + np.exp(z))


class OFUGLB(LogisticBandit):
    def __init__(self, param_norm_ub, arm_norm_ub, dim, failure_level, horizon, lazy_update_fr=1, plot_confidence=False, N_confidence=1000):
        """
        :param lazy_update_fr:  integer dictating the frequency at which to do the learning if we want the algo to be lazy (default: 1)
        """
        super().__init__(param_norm_ub, arm_norm_ub, dim, failure_level)
        self.name = 'OFUGLB'
        self.lazy_update_fr = lazy_update_fr
        # initialize some learning attributes
        self.theta_hat = np.random.normal(0, 1, (self.dim, 1))
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

        ## apparently, this is more inefficient than SciPy's SLSQP
        ## norm-constrained convex optim for MLE
        # theta_mle = cp.Variable((self.dim, 1))
        # theta_mle.value = self.theta_hat
        # constraint_mle = [cp.norm(theta_mle, 2) <= self.param_norm_ub]
        # objective_mle = cp.Minimize(self.neg_log_likelihood(theta_mle))
        # problem_mle = cp.Problem(objective_mle, constraint_mle)
        # problem_mle.solve(solver=cp.SCS, warm_start=True)
        # self.theta_hat = theta_mle.value

        obj = lambda theta: self.neg_log_likelihood_np(theta)
        cstrf_norm = lambda theta: np.linalg.norm(theta)
        constraint_norm = NonlinearConstraint(cstrf_norm, 0, self.param_norm_ub)
        opt = minimize(obj, x0=np.reshape(self.theta_hat, (-1,)), method='SLSQP', constraints=[constraint_norm])
        # , options={'maxiter': 20}
        self.theta_hat = opt.x
        # update counter
        self.ctr += 1

    def pull(self, arm_set):
        self.update_ucb_bonus()
        self.log_loss_hat = self.neg_log_likelihood_np(self.theta_hat)
        arm = np.reshape(arm_set.argmax(self.compute_optimistic_reward), (-1,))
        return arm

    # see Theorem 3.1
    def update_ucb_bonus(self):
        Lt = (1 + self.param_norm_ub / 2) * (len(self.rewards) - 1)
        self.ucb_bonus = np.log(1 / self.failure_level) + self.dim * np.log(max(math.e, 2 * math.e * self.param_norm_ub * Lt / self.dim))

    def compute_optimistic_reward(self, arm):
        arm_res = np.reshape(arm, (-1, 1))
        if self.ctr <= 1:
            res = np.random.normal(0, 1)
        else:
            ## apparently, this is more inefficient than SciPy's SLSQP
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

            obj = lambda theta: -np.sum(arm * theta)
            cstrf = lambda theta: self.neg_log_likelihood_np(theta) - self.log_loss_hat
            cstrf_norm = lambda theta: np.linalg.norm(theta, 2)
            constraint = NonlinearConstraint(cstrf, 0, self.ucb_bonus)
            constraint_norm = NonlinearConstraint(cstrf_norm, 0, self.param_norm_ub)
            opt = minimize(obj, x0=np.reshape(self.theta_hat, (-1,)), method='SLSQP', constraints=[constraint, constraint_norm])
            # , options={'maxiter': 20}
            res = np.sum(arm * opt.x)

            ## plot confidence set
            if self.plot and len(self.rewards) == self.T - 2:
                ## store data
                interact_rng = np.linspace(-self.param_norm_ub-0.5, self.param_norm_ub+0.5, self.N)
                x, y = np.meshgrid(interact_rng, interact_rng)
                f = lambda x, y: self.logistic_loss_seq(np.array([x, y])) - self.log_loss_hat
                z = (f(x, y) <= self.ucb_bonus) & (np.linalg.norm(np.array([x, y]), axis=0) <= self.param_norm_ub)
                z = z.astype(int)
                with open(f"S={self.param_norm_ub}/OFUGLB.npz", "wb") as file:
                    np.savez(file, theta_hat=self.theta_hat, x=x, y=y, z=z)
        return res

    def neg_log_likelihood(self, theta):
        """
        Computes the full log-loss estimated at theta
        """
        if len(self.rewards) == 0:
            return 0
        else:
            X = np.array(self.arms)
            r = np.array(self.rewards).reshape((-1, 1))
            theta = theta.reshape((-1, 1))
            # print(X.shape, r.shape)
            return cp.sum(cp.multiply(r, cp.logistic(-X @ theta)) + cp.multiply((1 - r), cp.logistic(X @ theta)))

    def neg_log_likelihood_np(self, theta):
        """
        Computes the full log-loss estimated at theta
        """
        if len(self.rewards) == 0:
            return 0
        else:
            X = np.array(self.arms)
            r = np.array(self.rewards).reshape((-1, 1))
            theta = theta.reshape((-1, 1))
            # print(X.shape, r.shape)
            return np.sum(r * logistic(-X @ theta) + (1 - r) * logistic(X @ theta))

    def logistic_loss_seq(self, theta):
        res = 0
        for s, r in enumerate(self.rewards):
            mu_s = 1 / (1 + np.exp(-np.tensordot(self.arms[s].reshape((self.dim,1)), theta, axes=([0], [0]))))
            # mu_s = np.clip(mu_s, 1e-12, 1 - 1e-12)
            if r == 0:
                res += -(1 - r) * np.log(1 - mu_s)
            else:
                res += -r * np.log(mu_s)
        return res.squeeze()





