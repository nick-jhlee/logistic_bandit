from logbexp.utils.utils import *


class LogisticBandit(object):
    """
    Class representing a base logistic bandit algorithm

    Attributes
    ----------
    param_norm_ub : float
        upper bound on the ell-two norm of theta_star (S)
    arm_norm_ub : float
        upper bound on the ell-two norm of any arm in the arm-set (L)
    dim : int
        problem dimension (d)
    failure_level: float
        failure level of the algorithm (delta)
    name : str
        algo name

    Methods
    -------
    pull(arm_set)
        play an arm within the given arm set

    learn(dataset)
        update internal parameters

    reset()
        reset attributes to initial value
    """

    def __init__(self, param_norm_ub, arm_norm_ub, dim, failure_level):
        self.param_norm_ub = param_norm_ub
        self.arm_norm_ub = arm_norm_ub
        self.dim = dim
        self.failure_level = failure_level
        self.name = None
        self.arms = []
        self.rewards = []

    def pull(self, arm_set):
        raise NotImplementedError

    def learn(self, arm, reward):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def neg_log_likelihood(self, theta, arms, rewards):
        """
        Computes the negative log likelihood at theta, over the prescribed arms and rewards
        """
        if len(rewards) == 0:
            return 0
        else:
            X = np.array(arms)
            r = np.array(rewards).reshape((-1, 1))
            theta = theta.reshape((-1, 1))
            # print(X.shape, r.shape)
            return - np.sum(r * np.log(sigmoid(X @ theta)) + (1 - r) * np.log(sigmoid(- X @ theta)))

    def neg_log_likelihood_J(self, theta, arms, rewards):
        """
        Derivative of neg_log_likelihood
        """
        if len(rewards) == 0:
            return np.zeros((self.dim, 1))
        else:
            X = np.array(arms)
            r = np.array(rewards).reshape((-1, 1))
            theta = theta.reshape((-1, 1))
            # print(X.shape, r.shape)
            return np.sum((sigmoid(X @ theta) - r) * X, axis=0).reshape((self.dim, 1))

    def neg_log_likelihood_full(self, theta):
        """
        Computes the full log-loss at theta
        """
        return self.neg_log_likelihood(theta, self.arms, self.rewards)

    def neg_log_likelihood_full_J(self, theta):
        """
        Derivative of neg_log_likelihood_full
        """
        return self.neg_log_likelihood_J(theta, self.arms, self.rewards)

    def neg_log_likelihood_plotting(self, grid):
        """
        Computes the full negative log likelihood at theta
        Taylor made for plotting
        grid : (2, N, N)
        """
        if len(self.rewards) == 0:
            return 0
        else:
            X = np.array(self.arms)
            tmp = np.einsum('td,dij->tij', X, grid)
            tmp1 = np.einsum('t,tij->ij', np.array(self.rewards), np.log(sigmoid(tmp)))
            tmp2 = np.einsum('t,tij->ij', (1 - np.array(self.rewards)), np.log(sigmoid(-tmp)))
            return - tmp1 - tmp2

    def logistic_loss_seq(self, theta):
        """
        For plotting the CS
        """
        res = 0
        for s, r in enumerate(self.rewards):
            mu_s = 1 / (1 + np.exp(-np.tensordot(self.arms[s].reshape((self.dim, 1)), theta, axes=([0], [0]))))
            # mu_s = np.clip(mu_s, 1e-12, 1 - 1e-12)
            if r == 0:
                res += -(1 - r) * np.log(1 - mu_s)
            else:
                res += -r * np.log(mu_s)
        return res.squeeze()
