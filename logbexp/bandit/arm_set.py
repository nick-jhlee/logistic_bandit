import numpy as np
from enum import Enum
import ipdb

"""
Arm-set class. 

Attributes
----------
arm_set_type : str
    the type of arm-set; options are ['fixed_discrete', 'tv_discrete', 'ball', 'movielens']
dim : int
    dimension
arm_set_size : int
    number of arms - does not matter for ball arm-set
arm_norm_ub : float
    upper-bound on the ell-two norm of the arms
Xu : np.ndarray
    user features
Xi : np.ndarray
    item features
rng : np.random.Generator
    random number generator
arm_option : str
    normalization option
user_size : int
    number of users
user_features : bool
    whether user features are used
"""


class AdmissibleArmSet(Enum):
    fxd = 'fixed_discrete'
    tvd = 'tv_discrete'
    ball = 'ball'
    movielens = 'movielens'
    
    @classmethod
    def from_string(self, str):
        return self._value2member_map_[str]

def is_perfect_square(num):
    sqrt_num = np.sqrt(num)
    return np.floor(sqrt_num) == np.ceil(sqrt_num) 

def project_onto_top_eigenvectors(X,k):
    me = np.mean(X, 0)
    centered = X - me 
    _, U = np.linalg.eigh(centered.T @ centered)
    return X @ U[:,-k:]



class ArmSet(object):
    def __init__(self, arm_set_type, dim, arm_set_size, arm_norm_ub, rng, Xu=None, Xi=None, arm_option=None, user_size=None, user_features=False):
        self.type = arm_set_type
        self.dim = dim
        self.arm_set_size = arm_set_size
        self.arm_norm_ub = arm_norm_ub
        self.arm_list = None
        self.user_features = user_features

        if user_features:
            assert Xu.shape[1] == Xi.shape[1]
            assert dim <= Xu.shape[1] ** 2
            assert is_perfect_square(dim)
            self.sqrt_dim = np.floor(np.sqrt(dim)).astype(int)
            self.user_size = user_size
            self.Xu = Xu[:,:self.sqrt_dim]
            self.Xi = Xi[:self.arm_set_size,:self.sqrt_dim]

            self.rng = rng
            # - NOTE when sqrt_dim = 2, Xu has norm of (max=0.630, min=0.006)
            # -                         Xi has norm of (max=0.383, min=0.049)
            # - FIXME project on to the top eigenvectors?

            if self.user_size is None:
                self.Xu = Xu[:, :self.sqrt_dim]
            else:
                idx = self.rng.permutation(Xu.shape[0])[:self.user_size]
                self.Xu = Xu[idx, :self.sqrt_dim]
            idx = self.rng.permutation(Xi.shape[0])[:self.arm_set_size]
            self.Xi = Xi[idx, :self.sqrt_dim]
            # self.Xi = Xi[:self.arm_set_size,:self.sqrt_dim]

            # - self.Xu = project_onto_top_eigenvectors(Xu, self.sqrt_dim)
            # - self.Xi = project_onto_top_eigenvectors(Xi, self.sqrt_dim)
            # rng = np.random.Generator(np.random.PCG64(120938))

            if arm_option == 'normalize':
                norms = np.array([np.linalg.norm(x) for x in self.Xu])
                self.Xu /= norms.reshape(-1, 1)

                norms = np.array([np.linalg.norm(x) for x in self.Xi])
                self.Xi /= norms.reshape(-1, 1)

    def generate_arm_list(self, seed=0):
        """
        Compute and stores the arm list.
        """
        if not self.type == AdmissibleArmSet.ball:
            if self.user_features:
                n_users = self.Xu.shape[0]
                i_u = self.rng.choice(range(n_users))
                u = self.Xu[i_u, :]
                self.arm_list = np.stack([np.outer(u, x).ravel() for x in self.Xi])
            else:
                rng = np.random.default_rng(seed)   # to keep the exp consistent across the repeats!
                u = rng.normal(0, 1, (self.arm_set_size, self.dim))
                norm = np.linalg.norm(u, axis=1)[:, None]
                r = rng.uniform(0, 1, (self.arm_set_size, 1)) ** (1.0 / self.dim)
                self.arm_list = r * u / norm

    def argmax(self, fun):
        """
        Find the arm which maximizes fun (only valid for finite arm sets).
        :param fun: function to maximize
        """
        if self.type == AdmissibleArmSet.ball:
            raise ValueError('argmax function is only compatible with finite arm sets')
        arm_and_values = list(zip(self.arm_list, [fun(a) for a in self.arm_list]))
        return max(arm_and_values, key=lambda x: x[1])[0]

    def random(self):
        """
        Draw and returns a random arm from arm_set.
        """
        if self.type == AdmissibleArmSet.ball:
            u = np.random.normal(0, 1, self.dim)
            norm = np.sqrt(np.sum(u ** 2))
            r = np.random.uniform(0, self.arm_norm_ub) ** (1.0 / self.dim)
            res = r * u / norm
        else:
            idx = np.random.randint(0, self.arm_set_size - 1, 1)
            res = self.arm_list[np.asscalar(idx)]
        return res


