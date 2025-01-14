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
import ipdb
from scipy.optimize import minimize

from logbexp.algorithms.logistic_bandit_algo import LogisticBandit


def mu(z):
    return 1 / (1 + np.exp(-z))

def mudot(z):
    return mu(z)*(1-mu(z))


def VAW_regularizer(arm, theta):
    """
    Computes the Vovk-Azoury-Warmuth predictor (regularizer) at theta for Bernoulli distribution
    see AIOLI of Jézéquel et al. (COLT '20)
    set lambda = 1
    """
    at = np.dot(arm, theta)
    return np.dot(theta, theta) + np.log(np.exp(at/2) + np.exp(-at/2))  # see Sec 3 of Bach (2010)


def VAW_regularizer_J(arm, theta):
    """
    Computes the gradient of VAW_regularizer
    """
    at = np.dot(arm, theta)
    return 2 * theta + (np.exp(at) - 1)/(2*(np.exp(at) + 1)) * arm


class EMK(LogisticBandit):
    def __init__(self, param_norm_ub, arm_norm_ub, dim, failure_level, horizon, lazy_update_fr=1, tol=1e-7):
        """
        :param lazy_update_fr:  integer dictating the frequency at which to do the learning if we want the algo to be lazy (default: 1)
        """
        super().__init__(param_norm_ub, arm_norm_ub, dim, failure_level)
        self.name = 'EMK'
        self.lazy_update_fr = lazy_update_fr
        # initialize some learning attributes
        self.theta_hat = np.zeros((self.dim, 1))
        self.ctr = 1
        self.T = horizon

        self.weighted_log_loss_hat = 0
        self.theta_hats = []
        self.weights = np.zeros((0,))
        self.L = 1 / 4  # for Bernoulli, L = 1/4
        self.l2reg = 1
        self.V_inv = self.l2reg**(-1) * np.eye(self.dim) # KJ should take self.l2reg**(-1)?
        # compute an upper bound on kappa (denoted as mu in Emmenegger et al. (2023))
        # KJ: actually, self.kappa is our typical kappa inverse (~ exp(-S))
        self.kappa = mudot(param_norm_ub)
        self.tol = tol

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

        # Sherman-Morrison formula
        V_inv_arm = self.V_inv @ arm
        self.V_inv -= (self.kappa * np.outer(V_inv_arm, V_inv_arm)
                       / (1 + self.kappa * (arm.T @ V_inv_arm)))

        # Vovk-Azoury-Warmuth predictor implemented with SLSQP
        ineq_cons = {'type': 'ineq',
                     'fun': lambda theta: self.param_norm_ub ** 2 - np.dot(theta, theta),
                     'jac': lambda theta: -2*theta}
        obj = lambda theta: self.neg_log_likelihood_full(theta) + VAW_regularizer(arm, theta)
        obj_J = lambda theta: self.neg_log_likelihood_full_J(theta) + VAW_regularizer_J(arm, theta)
        theta_hat_resized = np.reshape(self.theta_hat, (-1,))
        opt = minimize(obj, x0=theta_hat_resized, method='SLSQP', jac=obj_J,
                       constraints=ineq_cons, tol=self.tol)
        self.theta_hat = np.reshape(opt.x, (-1, 1))
        self.theta_hats.append(self.theta_hat)

        # update weighting (Theorem 2)
        bias2 = 2 * self.l2reg * (self.param_norm_ub ** 2) * (arm.T @ self.V_inv @ arm)
        weight = 1 / (1 + self.L * bias2)
        self.weights = np.concatenate((self.weights, [weight]))

        self.weighted_log_loss_hat += weight * self.neg_log_likelihood(self.theta_hat, arm, np.array([reward]))

        # update counter
        self.ctr += 1

    def pull(self, arm_set):
        arm = np.reshape(arm_set.argmax(self.compute_optimistic_reward), (-1,))
        return arm

    def compute_optimistic_reward(self, arm):
        if self.ctr <= 1:
            res = np.linalg.norm(arm)   # initially, choose the arm with the largest norm
        else:
            ## SLSQP
            obj = lambda theta: -np.sum(arm * theta)
            obj_J = lambda theta: -arm
            ineq_cons = {'type': 'ineq',
                         'fun': lambda theta: np.array([
                             np.log(1 / self.failure_level) - (
                                         self.neg_log_likelihood_sequential(theta) - self.weighted_log_loss_hat),
                             self.param_norm_ub ** 2 - np.dot(theta, theta)]),
                         'jac': lambda theta: - np.vstack((self.neg_log_likelihood_sequential_J(theta).T, 2 * theta))}
            theta_hat_resized = np.reshape(self.theta_hat, (-1,))
            opt = minimize(obj, x0=theta_hat_resized, method='SLSQP', jac=obj_J, constraints=ineq_cons, tol=self.tol)
            res = np.sum(arm * np.reshape(opt.x, (-1, 1)))

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

    ## Redefined to be adapted to the weighted, sequential setting!!
    def neg_log_likelihood_sequential(self, theta):
        """
        Computes the full, weighted negative log likelihood at theta
        """
        if len(self.rewards) == 0:
            return 0
        else:
            arms_theta = self.arms @ theta
            return - np.sum(self.weights * (self.rewards * np.log(mu(arms_theta)) + (1 - self.rewards) * np.log(mu(- arms_theta))))

    def neg_log_likelihood_sequential_J(self, theta):
        """
        Derivative of neg_log_likelihood_sequential
        """
        if len(self.rewards) == 0:
            return np.zeros((self.dim, 1))
        else:
            mus = np.reshape(mu(self.arms @ theta) - self.rewards, (-1, 1))
            weights_resized = np.reshape(self.weights, (-1, 1))
            return np.sum(weights_resized * (mus * self.arms), axis=0).reshape((self.dim, 1))

    def neg_log_likelihood_sequential_plotting(self, grid):
        """
        Computes the full, weighted negative log likelihood at theta
        Taylor made for plotting
        grid : (d, N, N)
        """
        if len(self.rewards) == 0:
            return 0
        else:
            r_weights1 = self.rewards * self.weights
            r_weights2 = (1 - self.rewards) * self.weights

            # Initialize the result arrays
            tmp1_sum = np.zeros((grid.shape[1], grid.shape[2]))
            tmp2_sum = np.zeros((grid.shape[1], grid.shape[2]))

            # Split arrays into chunks to include the remainder
            chunk_size = 100  # Adjust this based on your memory capacity
            num_sections = np.ceil(self.arms.shape[0] / chunk_size)
            arms_chunks = np.array_split(self.arms, num_sections)
            r_weights1_chunks = np.array_split(r_weights1, num_sections)
            r_weights2_chunks = np.array_split(r_weights2, num_sections)

            for arms_chunk, r_weights1_chunk, r_weights2_chunk in zip(arms_chunks, r_weights1_chunks, r_weights2_chunks):
                tmp_chunk = np.einsum('td,dij->tij', arms_chunk, grid)
                tmp1_chunk = np.einsum('t,tij->ij', r_weights1_chunk, np.log(mu(tmp_chunk)))
                tmp2_chunk = np.einsum('t,tij->ij', r_weights2_chunk, np.log(mu(-tmp_chunk)))

                tmp1_sum += tmp1_chunk
                tmp2_sum += tmp2_chunk

            return - tmp1_sum - tmp2_sum
