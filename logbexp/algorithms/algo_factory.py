from logbexp.algorithms.ecolog import EcoLog
from logbexp.algorithms.glm_ucb import GlmUCB
from logbexp.algorithms.gloc import Gloc
from logbexp.algorithms.logistic_ucb_1 import LogisticUCB1
from logbexp.algorithms.ol2m import Ol2m
from logbexp.algorithms.ofulogr import OFULogr
from logbexp.algorithms.ofulogplus import OFULogPlus
from logbexp.algorithms.ofuglb import OFUGLB
from logbexp.algorithms.ofuglb_e import OFUGLBe
from logbexp.algorithms.rs_glincb import RS_GLinCB
from logbexp.algorithms.emk import EMK

ALGOS = ['EMK', 'RS-GLinCB', 'OFUGLB-e', 'OFUGLB', 'OFULogPlus', 'adaECOLog', 'OFULog-r', 'LogUCB1', 'OL2M', 'GLOC', 'GLM-UCB']


def create_algo(config):
    """
    Creates algorithm from config.
    """
    algo = None
    if config["algo_name"] == 'GLM-UCB':
        algo = GlmUCB(param_norm_ub=config["param_norm_ub"],
                      arm_norm_ub=config["arm_norm_ub"],
                      dim=config["dim"],
                      failure_level=config["failure_level"])
    elif config["algo_name"] == 'LogUCB1':
        algo = LogisticUCB1(param_norm_ub=config["param_norm_ub"],
                            arm_norm_ub=config["arm_norm_ub"],
                            dim=config["dim"],
                            failure_level=config["failure_level"])
    elif config["algo_name"] == 'OFULog-r':
        algo = OFULogr(param_norm_ub=config["param_norm_ub"],
                       arm_norm_ub=config["arm_norm_ub"],
                       dim=config["dim"],
                       failure_level=config["failure_level"],
                       horizon=config["horizon"],
                       plot_confidence=config["plot_confidence"],
                       N_confidence=config["N_confidence"])
    elif config["algo_name"] == 'OL2M':
        algo = Ol2m(param_norm_ub=config["param_norm_ub"],
                    arm_norm_ub=config["arm_norm_ub"],
                    dim=config["dim"],
                    failure_level=config["failure_level"])
    elif config["algo_name"] == 'GLOC':
        algo = Gloc(param_norm_ub=config["param_norm_ub"],
                    arm_norm_ub=config["arm_norm_ub"],
                    dim=config["dim"],
                    failure_level=config["failure_level"])
    elif config["algo_name"] == 'adaECOLog':
        algo = EcoLog(param_norm_ub=config["param_norm_ub"],
                      arm_norm_ub=config["arm_norm_ub"],
                      dim=config["dim"],
                      failure_level=config["failure_level"],
                      horizon=config["horizon"],
                      plot_confidence=config["plot_confidence"],
                      N_confidence=config["N_confidence"],
                      arm_set_type=config["arm_set_type"])
    elif config["algo_name"] == 'OFULogPlus':
        algo = OFULogPlus(param_norm_ub=config["param_norm_ub"],
                          arm_norm_ub=config["arm_norm_ub"],
                          dim=config["dim"],
                          failure_level=config["failure_level"],
                          horizon=config["horizon"],
                          plot_confidence=config["plot_confidence"],
                          N_confidence=config["N_confidence"],
                      arm_set_type=config["arm_set_type"])
    elif config["algo_name"] == 'OFUGLB':
        algo = OFUGLB(param_norm_ub=config["param_norm_ub"],
                      arm_norm_ub=config["arm_norm_ub"],
                      dim=config["dim"],
                      failure_level=config["failure_level"],
                      horizon=config["horizon"],
                      plot_confidence=config["plot_confidence"],
                      N_confidence=config["N_confidence"],
                      arm_set_type=config["arm_set_type"])
    elif config["algo_name"] == 'OFUGLB-e':
        algo = OFUGLBe(param_norm_ub=config["param_norm_ub"],
                       arm_norm_ub=config["arm_norm_ub"],
                       dim=config["dim"],
                       failure_level=config["failure_level"],
                       horizon=config["horizon"],
                       plot_confidence=config["plot_confidence"],
                       N_confidence=config["N_confidence"],
                      arm_set_type=config["arm_set_type"])
    elif config["algo_name"] == 'RS-GLinCB':
        algo = RS_GLinCB(param_norm_ub=config["param_norm_ub"],
                       arm_norm_ub=config["arm_norm_ub"],
                       dim=config["dim"],
                       failure_level=config["failure_level"],
                       horizon=config["horizon"],
                       plot_confidence=config["plot_confidence"],
                       N_confidence=config["N_confidence"],
                      arm_set_type=config["arm_set_type"])
    elif config["algo_name"] == 'EMK':
        algo = EMK(param_norm_ub=config["param_norm_ub"],
                         arm_norm_ub=config["arm_norm_ub"],
                         dim=config["dim"],
                         failure_level=config["failure_level"],
                         horizon=config["horizon"],
                         plot_confidence=config["plot_confidence"],
                         N_confidence=config["N_confidence"],
                      arm_set_type=config["arm_set_type"])
    if algo is None:
        raise ValueError("Oops. The algorithm {} is not implemented. You must choose within ({})".format(
            config["algo_name"], ''.join(['\'' + elem + '\'' + ', ' for elem in ALGOS])))
    return algo
