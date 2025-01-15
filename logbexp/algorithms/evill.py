"""
Created on 2024.08.04
@author: DavidJanz, nicklee
Copied from https://github.com/DavidJanz/EVILL-code/tree/master with some modifications
Modification: replace IRLS with scipy.optimize.minimize (SLSQP) for optimization subroutines

Class for the EVILL of [Janz et al., AISTATS'24]. Inherits from the LogisticBandit class.

Here, for a fair comparison, we utilize the theoretically precise hyperparameters as in their Theorem 3 in Appendix E

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
import ipdb
from scipy.optimize import minimize

from logbexp.algorithms.logistic_bandit_algo import LogisticBandit
from logbexp.utils.optimization import fw_design



def mu(z):
    return 1 / (1 + np.exp(-z))

def dmu(z):
    return mu(z) * (1 - mu(z))


# TODO: change the hyperparameter as in Appendix E of Janz et al. (2024)
class EVILL(LogisticBandit):
    def __init__(self, param_norm_ub, arm_norm_ub, dim, failure_level, horizon, lazy_update_fr=1, tol=1e-7, arm_set_type=True):
        """
        :param lazy_update_fr:  integer dictating the frequency at which to do the learning if we want the algo to be lazy (default: 1)
        """
        super().__init__(param_norm_ub, arm_norm_ub, dim, failure_level)
        self.name = 'EVILL'
        self.lazy_update_fr = lazy_update_fr
        # initialize some learning attributes
        self.theta_hat = np.zeros((self.dim, 1)) # initialize at zero
        self.theta_perturbed = np.random.normal(0, 1, (self.dim, 1))
        self.ctr = 1
        self.T = horizon

        # hyperparameters (see Appendix E of Janz et al. (AISTATS '24))
        self.L = 1 / 4  # max of \dot{\mu}
        self.delta = failure_level / 3
        self.delta_prime = min((self.delta / self.T), 1/200)

        tmp = np.log(max(1/self.delta, np.e * np.sqrt(1 + self.T * self.L / dim)))
        self.l2reg = max(1, (2*dim / param_norm_ub) * tmp)
        self.gamma = np.sqrt(self.l2reg) * (0.5 + param_norm_ub) + (2*dim / np.sqrt(self.l2reg)) * tmp

        self.kappa = max(1, 2 + np.exp(self.param_norm_ub) + np.exp(-self.param_norm_ub))   # max(1, max_{|u| <= S} (\dot{\mu}(u))^{-1})
        C = np.sqrt(dim) + np.sqrt(2 * np.log(1 / self.delta_prime))
        Xi = np.sqrt(2) * (2*param_norm_ub + 0.5)
        D = Xi + Xi**2
        self.b = 1 / (22 * (1 + D * C**2) * C**2 * self.gamma * np.sqrt(self.kappa))
        self.tol = tol

        # for warmup
        self.V_tau = np.zeros((self.dim, self.dim))
        self.V_tau_inv = None
        self.V_invertible = False
        self.design = None

        # whether arm set is fixed or not
        self.arm_set_type = arm_set_type

    def reset(self):
        """
        Resets the underlying learning algorithm
        """
        self.theta_hat = np.zeros((self.dim, 1))
        self.ctr = 1
        self.arms = np.zeros((0, self.dim))
        self.rewards = np.zeros((0,))

    def learn(self, arm, reward):
        """
        Updates estimator.
        """
        self.arms = np.vstack((self.arms, arm))
        self.rewards = np.concatenate((self.rewards, [reward]))

        # original MLE (line 2)
        ineq_cons = {'type': 'ineq',
                        'fun': lambda theta: self.param_norm_ub ** 2 - np.dot(theta, theta),
                        'jac': lambda theta: -2 * theta
                    }
        obj = lambda theta: self.neg_log_likelihood_full(theta)
        obj_J = lambda theta: self.neg_log_likelihood_full_J(theta)
        theta_hat_resized = np.reshape(self.theta_hat, (-1,))
        opt = minimize(obj, x0=theta_hat_resized, method='SLSQP', jac=obj_J,
                        constraints=ineq_cons, tol=self.tol)
        self.theta_hat = np.reshape(opt.x, (-1, 1))
        # self.theta_hats.append(self.theta_hat)

        # sample random perturbation (line 3)
        zt = np.random.normal(0, 1, (self.dim, ))
        zt_ = np.random.normal(0, 1, (self.ctr - 1, ))

        # compute perturbation vector (line 4)
        # print(self.arms[0])
        # print(np.shape(np.vstack(self.arms)), np.shape(self.theta_hat), np.shape(zt_))
        wt = self.gamma * ( np.sqrt(self.l2reg) * zt + np.sum(np.sqrt(dmu(self.arms @ self.theta_hat)) * zt_))

        # compute perturbed MLE (line 5)
        ineq_cons = {'type': 'ineq',
                        'fun': lambda theta: self.param_norm_ub ** 2 - np.dot(theta, theta),
                        'jac': lambda theta: -2 *theta
                    }
        obj = lambda theta: self.neg_log_likelihood_full(theta) + np.dot(wt.T, theta)
        obj_J = lambda theta: self.neg_log_likelihood_full_J(theta) + wt
        theta_hat_resized = np.reshape(self.theta_hat, (-1,))
        opt = minimize(obj, x0=theta_hat_resized, method='SLSQP', jac=obj_J,
                        constraints=ineq_cons, tol=self.tol)
        self.theta_perturbed = np.reshape(opt.x, (-1, 1))

        # update counter
        self.ctr += 1

    def pull(self, arm_set):
        arm_list = arm_set.arm_list
        # if the arm-set is time-varying, then skip the warmup stage
        if self.arm_set_type in ["tv-discrete", "movielens"]:
            arm = np.reshape(arm_set.argmax(self.compute_optimistic_reward), (-1,))
            return arm
        # if the arm-set is fixed, then perform the theoretical EVILL
        else:
            # At t = 1, compute G-optimal design via Frank-Wolfe (thx to Brano for the codes!)
            if self.ctr == 1:
                self.design = fw_design(arm_list)
                # print(self.design)

            ## WARMUP via sampling from the G-optimal design until the criterion is satisfied
            ## See Appendix B of Janz et al. (AISTATS '24) and references therein
            if not self.V_invertible:
                # print(f"V_tau is not invertible, t={self.ctr}")
                # sample an index from design
                idx = np.random.choice(len(self.design), 1, p=self.design)[0]
                arm = np.reshape(arm_list[idx], (-1,))
                self.V_tau += np.outer(arm, arm)
                if np.linalg.det(self.V_tau) != 0:
                    self.V_tau_inv = np.linalg.inv(self.V_tau)
                    self.V_invertible = True
                return arm
            elif max([arm.T @ self.V_tau_inv @ arm for arm in arm_list]) > self.b:
                # print(f"warmup criterion not satisfied yet, t={self.ctr}")
                # sample an index from design
                idx = np.random.choice(len(self.design), 1, p=self.design)[0]
                arm = np.reshape(arm_list[idx], (-1,))
                self.V_tau += np.outer(arm, arm)
                self.V_tau_inv -= np.outer(self.V_tau_inv @ arm, self.V_tau_inv @ arm) / (1 + arm.T @ self.V_tau_inv @ arm)   # Sherman-Morrison formula
                return arm
            else:
                # print(f"Yay, t={self.ctr}")
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
        if self.ctr <= 1:
            return np.linalg.norm(arm)
        else:
            # return perturbed optimistic reward
            return np.dot(arm, self.theta_perturbed)


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