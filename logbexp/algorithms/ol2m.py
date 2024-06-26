
import numpy as np

from logbexp.algorithms.logistic_bandit_algo import LogisticBandit
from numpy.linalg import slogdet
from logbexp.utils.utils import sigmoid, weighted_norm, gaussian_sample_ellipsoid

"""
Class for the OL2M algorithm of [Zhang et al. 2016]. Inherits from the LogisticBandit class.

...

Additional Attributes
---------------------
l2_reg: float
    ell-two regularization of maximum-likelihood problem (lambda)
v_matrix: np.array(dim x dim)
    design matrix for Ol2m
v_matrix_inv: np.array(dim x dim)
    inverse design matrix for Ol2m
tetha: np.array(dim)
    estimation parameter
beta: float
    approximation of minimal variance (alternative computation for 1/kappa)
ctr : int
    counter 
"""


class Ol2m(LogisticBandit):
    def __init__(self, param_norm_ub, arm_norm_ub, dim, failure_level):
        super().__init__(param_norm_ub, arm_norm_ub, dim, failure_level)
        self.name = 'OL2M'
        self.l2reg = dim
        self.v_matrix = self.l2reg * np.eye(self.dim)
        self.v_matrix_inv = (1/self.l2reg)*np.eye(self.dim)
        self.theta = np.zeros((self.dim,))
        self.beta = 0.5 / (1 + np.exp(param_norm_ub))
        self.ctr = 1
        self.conf_radius = 0

    def reset(self):
        """

        Resets the underlying learning algorithm
        :return: None
        """
        self.v_matrix = self.l2reg * np.eye(self.dim)
        self.v_matrix_inv = (1 / self.l2reg) * np.eye(self.dim)
        self.theta = np.random.normal(0, 1, (self.dim,))
        self.ctr = 1

    def learn(self, arm, reward):
        current_grad = (sigmoid(np.dot(arm, self.theta)) - reward) * arm
        unprojected_estimate = self.theta - np.dot(self.v_matrix_inv, current_grad)
        # projection on ell-2 ball
        self.theta = self.param_norm_ub * unprojected_estimate / np.linalg.norm(unprojected_estimate)

    def pull(self, arm_set):
        # bonus-based version (strictly equivalent to param-based for this algo) of OL2M - see Sec 3.3 of Zhang et al (ICML'16)
        self.update_ucb_bonus()
        if not arm_set.type == 'ball':
            # find optimistic arm
            arm = np.reshape(arm_set.argmax(self.compute_optimistic_reward), (-1,))
        else:  # TS, only valid for unit ball arm-set
            param = gaussian_sample_ellipsoid(self.theta, self.v_matrix, self.conf_radius)
            arm = self.arm_norm_ub * param / np.linalg.norm(param)
        # update design matrix and inverse
        self.v_matrix += (self.beta / 2) * np.outer(arm, arm)
        self.v_matrix_inv += - (self.beta / 2) * np.dot(self.v_matrix_inv,
                                                        np.dot(np.outer(arm, arm), self.v_matrix_inv)) / (
                                     1 + (self.beta / 2) * np.dot(arm, np.dot(self.v_matrix_inv, arm)))
        self.ctr += 1
        return arm

    def update_ucb_bonus(self):
        """
        Update the ucb bonus function (cf. Thm 1 of [Zhang et al. 2016])
        :return:
        """
        tau = np.log(4*np.log(self.ctr+1) * self.ctr**2 / self.failure_level)
        res_square = 8*self.param_norm_ub + self.l2reg * self.param_norm_ub**2
        res_square += (8/self.beta + 16 * self.param_norm_ub / 3) * tau
        res_square += (2/self.beta) * (np.linalg.slogdet(self.v_matrix)[1] - self.dim*np.log(self.l2reg))
        self.conf_radius = np.sqrt(res_square)

    def compute_optimistic_reward(self, arm):
        """
        :param arm: np.array(dim)
        :return: the optimistic reward associated to arm
        """
        norm = weighted_norm(arm, self.v_matrix_inv)
        pred_reward = sigmoid(np.sum(self.theta*arm))
        bonus = self.conf_radius * norm
        return pred_reward+bonus
