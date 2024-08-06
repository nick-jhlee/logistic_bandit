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
    arm_res = np.reshape(arm, (-1, 1))
    theta_res = np.reshape(theta, (-1, 1))
    return 2 * theta_res + mu(np.dot(arm, theta)) * arm_res


class EMK(LogisticBandit):
    def __init__(self, param_norm_ub, arm_norm_ub, dim, failure_level, horizon, arm_set_type="tv_discrete", lazy_update_fr=1, plot_confidence=False,
                 N_confidence=1000):
        """
        :param lazy_update_fr:  integer dictating the frequency at which to do the learning if we want the algo to be lazy (default: 1)
        """
        super().__init__(param_norm_ub, arm_norm_ub, dim, failure_level)
        self.name = 'EMK'
        self.arm_set_type = arm_set_type
        self.lazy_update_fr = lazy_update_fr
        # initialize some learning attributes
        self.arms = []
        self.rewards = []
        self.theta_hat = np.random.normal(0, 1, (self.dim, 1))
        self.ctr = 1
        self.log_loss_hat = 0
        self.plot = plot_confidence
        self.N = N_confidence
        self.T = horizon

        self.weighted_log_hat = 0
        self.theta_hats = []
        self.weights = []
        self.L = 1 / 4  # for Bernoulli, L = 1/4
        self.l2reg = 1
        self.V_inv = self.l2reg * np.eye(self.dim)
        # compute an upper bound on kappa (denoted as mu in Emmenegger et al. (2023))
        self.kappa = 3 + np.exp(param_norm_ub)

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

        # Sherman-Morrison formula
        self.V_inv -= (self.kappa * np.outer(self.V_inv @ arm, self.V_inv @ arm)
                       / (1 + self.kappa * arm.T @ self.V_inv @ arm))

        # Vovk-Azoury-Warmuth predictor (see their Theorem 5), implemented with SLSQP
        ineq_cons = {'type': 'ineq',
                     'fun': lambda theta: np.array([self.param_norm_ub ** 2 - np.dot(theta, theta)]),
                     'jac': lambda theta: 2 * np.array([- theta])}
        obj = lambda theta: self.neg_log_likelihood_full(theta) + VAW_regularizer(arm, theta)
        obj_J = lambda theta: self.neg_log_likelihood_full_J(theta) + VAW_regularizer_J(arm, theta)
        opt = minimize(obj, x0=np.reshape(self.theta_hat, (-1,)), method='SLSQP', jac=obj_J,
                       constraints=ineq_cons)
        self.theta_hat = opt.x
        self.theta_hats.append(self.theta_hat)

        # update weighting (Theorem 2)
        bias2 = 2 * self.l2reg * self.param_norm_ub ** 2 * arm.T @ self.V_inv @ arm
        self.weights.append(1 / (1 + self.L * bias2))

        self.weighted_log_hat += self.weights[-1] * self.neg_log_likelihood(self.theta_hat, [arm], [reward])

        # update counter
        self.ctr += 1

    def pull(self, arm_set):
        arm = np.reshape(arm_set.argmax(self.compute_optimistic_reward), (-1,))
        return arm

    def compute_optimistic_reward(self, arm):
        if self.ctr <= 1:
            res = np.random.normal(0, 1)
        else:
            ## SLSQP
            obj = lambda theta: -np.sum(arm * theta)
            obj_J = lambda theta: -arm
            ineq_cons = {'type': 'ineq',
                         'fun': lambda theta: np.array([
                             np.log(1 / self.failure_level) - (self.neg_log_likelihood_sequential(theta) - self.weighted_log_hat),
                             self.param_norm_ub ** 2 - np.dot(theta, theta)]),
                         'jac': lambda theta: - np.vstack((self.neg_log_likelihood_sequential_J(theta).T, 2 * theta))}
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
                X, Y = np.meshgrid(interact_rng, interact_rng)
                f = lambda x, y: self.neg_log_likelihood_sequential_plotting(np.array([x, y])) - self.weighted_log_hat
                Z = ((f(X, Y) <= np.log(1 / self.failure_level))
                     & (np.linalg.norm(np.array([X, Y]), axis=0) <= self.param_norm_ub))
                Z = Z.astype(int)
                self.save_npz(X, Y, Z, self.theta_hat)
        return res
    

    ## Redefined to be adapted to the weighted, sequential setting!!
    def neg_log_likelihood_sequential(self, theta):
        """
        Computes the full, weighted negative log likelihood at theta
        """
        if len(self.rewards) == 0:
            return 0
        else:
            X = np.array(self.arms)
            r = np.array(self.rewards).reshape((-1, 1))
            theta = theta.reshape((-1, 1))
            weights = np.array(self.weights).reshape((-1, 1))
            # print(X.shape, r.shape)
            return - np.sum(weights * (r * np.log(mu(X @ theta)) + (1 - r) * np.log(mu(- X @ theta))))

    def neg_log_likelihood_sequential_J(self, theta):
        """
        Derivative of neg_log_likelihood_sequential
        """
        if len(self.rewards) == 0:
            return np.zeros((self.dim, 1))
        else:
            X = np.array(self.arms)
            r = np.array(self.rewards).reshape((-1, 1))
            theta = theta.reshape((-1, 1))
            weights = np.array(self.weights).reshape((-1, 1))
            # print(X.shape, r.shape)
            return np.sum(weights * ((mu(X @ theta) - r) * X), axis=0).reshape((self.dim, 1))

    def neg_log_likelihood_sequential_plotting(self, grid):
        """
        Computes the full, weighted negative log likelihood at theta
        Taylor made for plotting
        grid : (d, N, N)
        """
        if len(self.rewards) == 0:
            return 0
        else:
            X = np.array(self.arms)
            tmp = np.einsum('td,dij->tij', X, grid)
            r_weights1 = np.array(self.rewards) * np.array(self.weights)
            tmp1 = np.einsum('t,tij->ij', r_weights1, np.log(mu(tmp)))
            r_weights2 = (1 - np.array(self.rewards)) * np.array(self.weights)
            tmp2 = np.einsum('t,tij->ij', r_weights2, np.log(mu(-tmp)))
            return - tmp1 - tmp2