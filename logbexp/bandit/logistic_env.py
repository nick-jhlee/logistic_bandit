import numpy as np
import os
import ipdb

from logbexp.bandit.logistic_oracle import LogisticOracle
from logbexp.bandit.arm_set import AdmissibleArmSet, ArmSet
from logbexp.utils.utils import sigmoid, dsigmoid
# from kjunutils3_v3_ import LoadPickleGzip

"""
Logistic Bandit Environment class

Attributes
---------
oracle : LogisticOracle
arm_set : ArmSet
rng : numpy.random.Generator
"""

class LogisticBanditEnv(object):
    def __init__(self, theta_star, arm_set, rng):
        self.oracle = LogisticOracle(theta_star, rng)
        self.arm_set = arm_set
        self.arm_set.generate_arm_list()
        self.count = 1
        self.rng = rng

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
        if self.arm_set.type in [AdmissibleArmSet.tvd, AdmissibleArmSet.movielens] :
            self.arm_set.generate_arm_list(self.count) # self.count becomes the random seed!
            self.count += 1
        return reward, regret, kappa_inv

    def get_best_arm(self):
        """
        Returns the expected reward of the best arm.
        """
        if self.arm_set.type == AdmissibleArmSet.ball:
            # raise NotImplementedError()
            best_arm = self.arm_set.arm_norm_ub * self.oracle.theta_star / np.linalg.norm(self.oracle.theta_star)
        else:
            vals = self.arm_set.arm_list @ self.oracle.theta_star
            argmax = np.argmax(vals)
            best_arm = self.arm_set.arm_list[argmax]
            # perf_fun = lambda x: np.sum(x*self.oracle.theta_star)
            # best_arm = self.arm_set.argmax(perf_fun)
        return best_arm


def create_env(config, seed):
    #- one common rng for both LogisticBanditEnv and ArmSet
    rng = np.random.Generator(np.random.PCG64(seed))

    theta_star = config["theta_star"]
    arm_set_type = AdmissibleArmSet(config['arm_set_type'])

    arm_set_size = config["arm_set_size"]
    arm_norm_ub = config["arm_norm_ub"]

    if arm_set_type == AdmissibleArmSet.movielens:
        here = os.path.dirname(os.path.abspath(__file__))
        data = np.load(os.path.join(here, '../..', 'data/out_pu_qi.npz'))
        Xu = data["pu"]
        Xi = data["qi"]
        arm_set = ArmSet(arm_set_type, len(theta_star), arm_set_size, arm_norm_ub, Xu, Xi, rng, arm_option=config['arm_option'], user_features=True)
    else:
        # raise NotImplementedError()
        arm_set = ArmSet(arm_set_type, len(theta_star), arm_set_size, arm_norm_ub, rng)

    return LogisticBanditEnv(theta_star, arm_set, rng)
