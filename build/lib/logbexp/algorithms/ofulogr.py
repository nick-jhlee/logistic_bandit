import numpy as np

from logbexp.algorithms.logistic_bandit_algo import LogisticBandit
from numpy.linalg import slogdet
from logbexp.utils.utils import sigmoid
from scipy.optimize import minimize, NonlinearConstraint
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


class OFULogr(LogisticBandit):
    def __init__(self, param_norm_ub, arm_norm_ub, dim, failure_level, lazy_update_fr=1, plot_confidence=False, N_confidence=500):
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
        self.plot = plot_confidence
        self.N = N_confidence

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

        # learn the m.l.e by iterative approach (a few steps of Newton descent)
        self.l2reg = self.dim * np.log(2 + len(self.rewards))
        if self.ctr % self.lazy_update_fr == 0 or len(self.rewards) < 200:
            ## scipy
            obj = lambda theta: self.logistic_loss(theta)
            opt = minimize(obj, x0=self.theta_hat, method='SLSQP')
            self.theta_hat = opt.x

            ## previously: learn the m.l.e by iterative approach (a few steps of Newton descent)
            # theta_hat = self.theta_hat
            # for _ in range(5):
            #     coeffs = sigmoid(np.dot(self.arms, theta_hat)[:, None])
            #     y = coeffs - np.array(self.rewards)[:, None]
            #     grad = self.l2reg * theta_hat + np.sum(y * self.arms, axis=0)
            #     hessian = np.dot(np.array(self.arms).T,
            #                      coeffs * (1 - coeffs) * np.array(self.arms)) + self.l2reg * np.eye(self.dim)
            #     theta_hat -= np.linalg.solve(hessian, grad)
            # self.theta_hat = theta_hat
            # self.hessian_matrix = hessian
        # update counter
        self.ctr += 1

    def pull(self, arm_set):
        self.update_ucb_bonus()
        self.log_loss_hat = self.logistic_loss(self.theta_hat)
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
            cstrf = lambda theta: self.logistic_loss(theta) - self.log_loss_hat
            cstrf_norm = lambda theta: np.linalg.norm(theta)
            constraint = NonlinearConstraint(cstrf, 0, self.ucb_bonus)
            constraint_norm = NonlinearConstraint(cstrf_norm, 0, self.param_norm_ub)
            opt = minimize(obj, x0=self.theta_hat, method='SLSQP', constraints=[constraint, constraint_norm])
            # options={'maxiter': 20}
            res = np.sum(arm * opt.x)

            ## plot confidence set
            if self.plot and len(self.rewards) == 4000:
                ## store data
                interact_rng = np.linspace(-self.param_norm_ub-0.5, self.param_norm_ub+0.5, self.N)
                x, y = np.meshgrid(interact_rng, interact_rng)
                f = lambda x, y: self.logistic_loss_seq(np.array([x, y])) - self.log_loss_hat
                z = (f(x, y) <= self.ucb_bonus) & (np.linalg.norm(np.array([x, y]), axis=0) <= self.param_norm_ub)
                z = z.astype(int)
                np.savez(f"S={self.param_norm_ub}/OFULogr.npz", x=x, y=y, z=z, theta_hat=self.theta_hat)
        return res

    def logistic_loss(self, theta):
        """
        Computes the full log-loss estimated at theta
        """
        res = self.l2reg / 2 * np.linalg.norm(theta)**2
        if len(self.rewards) > 0:
            coeffs = np.clip(sigmoid(np.dot(self.arms, theta)[:, None]), 1e-12, 1-1e-12)
            res += -np.sum(np.array(self.rewards)[:, None] * np.log(coeffs / (1 - coeffs)) + np.log(1 - coeffs))
        return res

    def logistic_loss_seq(self, theta):
        res = self.l2reg / 2 * np.linalg.norm(theta, axis=0)**2
        res = res.reshape((1, self.N, self.N))
        for s, r in enumerate(self.rewards):
            mu_s = 1 / (1 + np.exp(-np.tensordot(self.arms[s].reshape((self.dim,1)), theta, axes=([0], [0]))))
            mu_s = np.clip(mu_s, 1e-12, 1 - 1e-12)
            if r == 0:
                res += -(1 - r) * np.log(1 - mu_s)
            else:
                res += -r * np.log(mu_s)
        return res.squeeze()