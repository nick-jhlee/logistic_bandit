import numpy as np

from logbexp.algorithms.logistic_bandit_algo import LogisticBandit
from numpy.linalg import slogdet
from scipy.optimize import minimize
from scipy.stats import chi2

"""
Class for the OFULog-r algorithm of [Abeille et al. 2021]. Inherits from the LogisticBandit class.

Additional Attributes
---------------------
lazy_update_fr : int
    integer dictating the frequency at which to do the learning if we want the algo to be lazy
l2_reg: float
    ell-two regularization of maximum-likelihood problem (lambda)
hessian_matrix: np.array(dim x dim)
    hessian of the log-loss at current estimation (H_t)   
theta_hat : np.array(dim)
    maximum-likelihood estimator
log_loss_hat : float
    log-loss at current estimate theta_hat
ctr : int
    counter for lazy updates
"""


def logistic(z):
    return np.log(1 + np.exp(z))


class OFULogr(LogisticBandit):
    def __init__(self, param_norm_ub, arm_norm_ub, dim, failure_level, horizon, lazy_update_fr=1):
        """
        :param lazy_update_fr:  integer dictating the frequency at which to do the learning if we want the algo to be lazy (default: 1)
        """
        super().__init__(param_norm_ub, arm_norm_ub, dim, failure_level)
        self.name = 'OFULog-r'
        self.lazy_update_fr = lazy_update_fr
        # initialize some learning attributes
        self.l2reg = self.dim
        self.hessian_matrix = self.l2reg * np.eye(self.dim)
        self.theta_hat = np.random.normal(0, 1, (self.dim,))
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
        self.hessian_matrix = self.l2reg * np.eye(self.dim)
        self.theta_hat = np.random.normal(0, 1, (self.dim,))
        self.ctr = 1
        self.arms = []
        self.rewards = []

    def learn(self, arm, reward):
        """
        Updates estimator.
        """
        self.arms.append(arm)
        self.rewards.append(reward)

        self.l2reg = self.dim * np.log(2 + len(self.rewards))

        ## SLSQP for regularized MLE
        opt = minimize(self.neg_regularized_log_likelihood_full, x0=np.reshape(self.theta_hat, (-1,)), method='SLSQP',
                       jac=self.neg_regularized_log_likelihood_full_J)
        self.theta_hat = opt.x
        
        # update counter
        self.ctr += 1

    def pull(self, arm_set):
        self.update_ucb_bonus()
        self.log_loss_hat = self.neg_regularized_log_likelihood_full(self.theta_hat)
        arm = np.reshape(arm_set.argmax(self.compute_optimistic_reward), (-1,))
        return arm

    def update_ucb_bonus(self):
        """
        Updates the ucb bonus function (refined concentration result from Faury et al. 2020)
        """
        _, logdet = slogdet(self.hessian_matrix)
        gamma_1 = np.sqrt(self.l2reg)*(0.5 + self.param_norm_ub) + (2 / np.sqrt(self.l2reg)) \
                  * (np.log(1 / self.failure_level) + 0.5 * logdet - 0.5 * self.dim * np.log(self.l2reg) +
                     np.log(chi2(self.dim).cdf(2 * self.l2reg) / chi2(self.dim).cdf(self.l2reg)))
        gamma_2 = np.sqrt(self.l2reg)*self.param_norm_ub + np.log(1 / self.failure_level) \
                  + np.log(chi2(self.dim).cdf(2 * self.l2reg) / chi2(self.dim).cdf(self.l2reg)) \
                  + 0.5 * logdet - 0.5 * self.dim * np.log(self.l2reg)
        gamma = np.min([gamma_1, gamma_2])
        res = (gamma + gamma ** 2 / self.l2reg) ** 2
        self.ucb_bonus = res

    def compute_optimistic_reward(self, arm):
        """
        Planning according to Algo. 2 of Abeille et al. 2021
        """
        if self.ctr == 1:
            res = np.random.normal(0, 1)
        else:
            obj = lambda theta: -np.sum(arm * theta)
            obj_J = lambda theta: -arm
            ineq_cons = {'type': 'ineq',
                         'fun': lambda theta: np.array([
                             self.ucb_bonus - (self.neg_regularized_log_likelihood_full(theta) - self.log_loss_hat)
                             ]),
                         'jac': lambda theta: - self.neg_regularized_log_likelihood_full_J(theta).T}
            opt = minimize(obj, x0=self.theta_hat, method='SLSQP', jac=obj_J, constraints=ineq_cons)
            res = np.sum(arm * opt.x)
        return res

    def neg_regularized_log_likelihood_full(self, theta):
        """
        Computes the negative regularized log-likelihood
        """
        return self.neg_log_likelihood_full(theta) + (self.l2reg / 2) * np.dot(theta, theta)
    
    def neg_regularized_log_likelihood_full_J(self, theta):
        """
        Computes the gradient of the negative regularized log-likelihood
        """
        return self.neg_log_likelihood_full_J(theta) + self.l2reg * theta