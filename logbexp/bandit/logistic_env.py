import numpy as np

from logbexp.bandit.logistic_oracle import LogisticOracle
from logbexp.bandit.arm_set import AdmissibleArmSet, ArmSet

"""
Logistic Bandit Environment class

Attributes
---------
oracle : LogisticOracle
arm_set : ArmSet
"""


class LogisticBanditEnv(object):
    def __init__(self, theta_star, arm_set_type, arm_set_size, arm_norm_ub):
        self.oracle = LogisticOracle(theta_star)
        self.arm_set = ArmSet(arm_set_type, len(theta_star), arm_set_size, arm_norm_ub)
        self.arm_set.generate_arm_list()
        self.count = 1

    def interact(self, arm):
        """
        Returns the reward obtained after playing arm and the instantaneous pseudo-regret.
        (change by junghyun) also return kappa_t^{-1} at time t
        """
        reward = self.oracle.pull(arm)

        best_arm = self.get_best_arm()
        regret = self.oracle.expected_reward(best_arm) - self.oracle.expected_reward(arm)
        kappa_inv = dsigmoid(np.dot(best_arm, self.oracle.theta_star))

        # regenerates arm-set if the arm-set is time-varying.
        if self.arm_set.type == AdmissibleArmSet.tvd:
            self.arm_set.generate_arm_list(self.count)
            self.count += 1
        return reward, regret, kappa_inv

    def get_best_arm(self):
        """
        Returns the expected reward of the best arm.
        """
        if self.arm_set.type == AdmissibleArmSet.ball:
            best_arm = self.arm_set.arm_norm_ub * self.oracle.theta_star / np.linalg.norm(self.oracle.theta_star)
        else:
            perf_fun = lambda x: np.sum(x*self.oracle.theta_star)
            best_arm = self.arm_set.argmax(perf_fun)
        return best_arm


def create_env(config):
    theta_star = config["theta_star"]

    # test if arm_set_type is admissible
    try:
        arm_set_type = AdmissibleArmSet(config["arm_set_type"])
    except ValueError as e:
        raise ValueError('Oops. The arm-set \'{}\' is not admissible. It must belong to ({})'.format(
            config["arm_set_type"],
            ''.join(['\''+entry.value+'\'' + ',' for entry in AdmissibleArmSet]))) from e

    arm_set_size = config["arm_set_size"]
    arm_norm_ub = config["arm_norm_ub"]
    return LogisticBanditEnv(theta_star, arm_set_type, arm_set_size, arm_norm_ub)


def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))