"""
Created on 2024.08.04
@author: DavidJanz, nicklee
Copied from https://github.com/DavidJanz/EVILL-code/tree/master with some modifications
Modification: replace IRLS with scipy.optimize.minimize (SLSQP) for optimization subroutines

Class for the EVILL of [Janz et al., AISTATS'24]. Inherits from the LogisticBandit class.

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

def dmu(z):
    return mu(z) * (1 - mu(z))


class EVILL(LogisticBandit):
    def __init__(self, param_norm_ub, arm_norm_ub, dim, failure_level, horizon, lazy_update_fr=1):
        """
        :param lazy_update_fr:  integer dictating the frequency at which to do the learning if we want the algo to be lazy (default: 1)
        """
        super().__init__(param_norm_ub, arm_norm_ub, dim, failure_level)
        self.name = 'EVILL'
        self.lazy_update_fr = lazy_update_fr
        # initialize some learning attributes
        self.arms = []
        self.rewards = []
        self.theta_hat = np.random.normal(0, 1, (self.dim, 1))
        self.theta_perturbed = np.random.normal(0, 1, (self.dim, 1))
        self.ctr = 1
        self.T = horizon

        # hyperparameters
        self.tau = 20  ## warmup stage, length set to number of arms
        self.l2reg = 1.0    ## regularisation parameter
        self.a = 1  ## perturbation scale


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

        # update counter
        self.ctr += 1

    def pull(self, arm_set):
        # Warmup stage
        if self.ctr <= self.tau:
            return arm_set.arm_list[self.ctr % arm_set.length]
        else:
            arm = np.reshape(arm_set.argmax(self.compute_optimistic_reward), (-1,))
            return arm

    def compute_optimistic_reward(self, arm):
        # mean_theta, _ = irls(self.theta, self.arms, self.arm_outer_prods, self.num_pos,
        #                      self.num_neg, regularisation=1.0)
        #
        # u = self.arms @ mean_theta
        # z = self.a * np.sqrt(variance(u)) * np.sqrt(self.num_pulls) * np.random.randn(self.K)
        # y = np.random.randn(self.d) * self.a
        #
        # theta, _ = irls(self.theta, self.arms, self.arm_outer_prods, self.num_pos, self.num_neg,
        #                 regularisation=1.0, pre_perturbation=z, post_perturbation=y)
        #
        # mu = clipped_sigmoid(self.arms @ theta)
        if self.ctr <= self.tau:
            return 0
        else:
            # original MLE (line 2)
            ineq_cons = {'type': 'ineq',
                         'fun': lambda theta: np.array([self.param_norm_ub ** 2 - np.dot(theta, theta)]),
                         'jac': lambda theta: 2 * np.array([- theta])}
            obj = lambda theta: self.neg_log_likelihood_full(theta)
            obj_J = lambda theta: self.neg_log_likelihood_full_J(theta)
            opt = minimize(obj, x0=np.reshape(self.theta_hat, (-1,)), method='SLSQP', jac=obj_J,
                           constraints=ineq_cons)
            self.theta_hat = opt.x.reshape(-1, 1)
            # self.theta_hats.append(self.theta_hat)

            # sample random perturbation (line 3)
            zt = np.random.normal(0, 1, (self.dim, 1))
            zt_ = np.random.normal(0, 1, (self.ctr - 1, 1))

            # compute perturbation vector (line 4)
            # print(self.arms[0])
            # print(np.shape(np.vstack(self.arms)), np.shape(self.theta_hat), np.shape(zt_))
            wt = self.a * ( np.sqrt(self.l2reg) * zt + np.sum(np.sqrt(dmu(np.vstack(self.arms) @ self.theta_hat)) * zt_))

            # compute perturbed MLE (line 5)
            ineq_cons = {'type': 'ineq',
                         'fun': lambda theta: np.array([self.param_norm_ub ** 2 - np.dot(theta, theta)]),
                         'jac': lambda theta: 2 * np.array([- theta])}
            obj = lambda theta: self.neg_log_likelihood_full(theta) + np.dot(wt.T, theta)
            obj_J = lambda theta: self.neg_log_likelihood_full_J(theta) + wt
            opt = minimize(obj, x0=np.reshape(self.theta_hat, (-1,)), method='SLSQP', jac=obj_J,
                           constraints=ineq_cons)
            self.theta_perturbed = opt.x

            # return perturbed optimistic reward
            return mu(np.dot(arm, self.theta_perturbed))


    # # from DavidJanz
    # def clipped_sigmoid(x):
    #     x[x > 36] = 36
    #     x[x < -36] = 36
    #     return 1 / (1 + np.exp(-x))
    #
    # # from DavidJanz
    # def variance(x):
    #     return clipped_sigmoid(x) * (1 - clipped_sigmoid(x))
    # def irls(self, theta, arms, arm_outer_prods, num_pos, num_neg, regularisation=1.0, pre_perturbation=0.0,
    #          post_perturbation=0.0, num_iter=1000, tol=1e-8):
    #     """
    #     Iterative reweighted least squares for Bayesian logistic regression. See Sections 4.3.3 and
    #      4.5.1 in
    #         Bishop, Christopher M., and Nasser M. Nasrabadi. Pattern Recognition and Machine Learning.
    #         Vol. 4. No. 4. New York: Springer, 2006.
    #
    #     Returns: estimate of parameters and gram matrix
    #     """
    #     arms = np.copy(arms)
    #     theta = np.copy(theta)
    #     _, d = arms.shape
    #     gram = np.eye(d)
    #
    #     for i in range(num_iter):
    #         theta_old = np.copy(theta)
    #
    #         arms_theta_prod = arms @ theta
    #         means = clipped_sigmoid(arms_theta_prod)
    #         num_pulls = num_pos + num_neg
    #         gram = (np.tensordot(variance(arms_theta_prod) * num_pulls, arm_outer_prods,
    #                              axes=([0], [0])) + regularisation * np.eye(d))
    #         Rz = variance(
    #             arms_theta_prod) * num_pulls * arms_theta_prod + num_pos - num_pulls * means + pre_perturbation
    #         theta = np.linalg.solve(gram, arms.T @ Rz + post_perturbation)
    #
    #         if np.linalg.norm(theta - theta_old) < tol:
    #             break
    #     return theta, gram