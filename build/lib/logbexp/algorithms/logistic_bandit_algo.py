from logbexp.utils.utils import *
import ipdb


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
        self.arm_set_type = None
        # history of arms and rewards
        self.arms = np.zeros((0, self.dim))
        self.rewards = np.zeros((0,))

    def pull(self, arm_set):
        raise NotImplementedError

    def learn(self, arm, reward):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def gradient(self, theta, l2reg=0):
        """
        Computes the gradient of the negative log-likelihood at theta
        """
        mu_dots = dsigmoid(self.arms @ theta)
        return self.arms.T @ (mu_dots - self.rewards) + l2reg * theta

    def hessian(self, theta, l2reg=0):
        """
        Computes the Hessian of the negative log-likelihood at theta
        """
        mu_dots = np.reshape(dsigmoid(self.arms @ theta), (-1, 1))
        return self.arms.T @ (mu_dots * self.arms) + l2reg * np.eye(self.dim)

    def neg_log_likelihood(self, theta, arms, rewards):
        """
        """
        if len(rewards) == 0:
            return 0
        else:
            arms_theta = arms @ theta
            return - np.sum(rewards * np.log(sigmoid(arms_theta)) + (1 - rewards) * np.log(sigmoid(- arms_theta)))

    def neg_log_likelihood_J(self, theta, arms, rewards):
        """
        Derivative of neg_log_likelihood
        """
        if len(rewards) == 0:
            return np.zeros(self.dim)
        else:
            return arms.T @ (sigmoid(arms @ theta) - rewards)

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
        grid : (d, N, N)
        """
        if len(self.rewards) == 0:
            return 0
        else:
            # Initialize the result arrays
            tmp1_sum = np.zeros((grid.shape[1], grid.shape[2]))
            tmp2_sum = np.zeros((grid.shape[1], grid.shape[2]))

            # Split arrays into chunks to include the remainder
            chunk_size = 100  # Adjust this based on your memory capacity
            num_sections = np.ceil(self.arms.shape[0] / chunk_size)
            arms_chunks = np.array_split(self.arms, num_sections)
            rewards_chunks = np.array_split(self.rewards, num_sections)

            for arms_chunk, rewards_chunk in zip(arms_chunks, rewards_chunks):
                tmp_chunk = np.einsum('td,dij->tij', arms_chunk, grid)
                tmp1_chunk = np.einsum('t,tij->ij', rewards_chunk, np.log(sigmoid(tmp_chunk)))
                tmp2_chunk = np.einsum('t,tij->ij', (1 - rewards_chunk), np.log(sigmoid(-tmp_chunk)))

                tmp1_sum += tmp1_chunk
                tmp2_sum += tmp2_chunk

            return - tmp1_sum - tmp2_sum