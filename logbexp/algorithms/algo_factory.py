from logbexp.algorithms.ecolog import EcoLog
from logbexp.algorithms.glm_ucb import GlmUCB
from logbexp.algorithms.gloc import Gloc
from logbexp.algorithms.logistic_ucb_1 import LogisticUCB1
from logbexp.algorithms.ol2m import Ol2m
from logbexp.algorithms.ofulogr import OFULogr
from logbexp.algorithms.ofulogplus import OFULogPlus
from logbexp.algorithms.ofuglb import OFUGLB
from logbexp.algorithms.ofuglb_e import OFUGLBe
from logbexp.algorithms.evill import EVILL
# from logbexp.algorithms.ofuglb_ball_e import OFUGLBballe
from logbexp.algorithms.rs_glincb import RS_GLinCB
from logbexp.algorithms.emk import EMK

ALGOS = ['OL2M', 'LogUCB1', 'OFULog-r', 'adaECOLog', 'OFULogPlus', 'GLM-UCB', 'RS-GLinCB',
         'GLOC', 'EMK', 'EVILL', 'OFUGLB-e', 'OFUGLB']


def create_algo(config):
    """
    Creates algorithm from config.
    """
    algo = None
    if config["algo_name"] == 'GLM-UCB':
        algo = GlmUCB(param_norm_ub=config["param_norm_ub"],
                      arm_norm_ub=config["arm_norm_ub"],
                      dim=config["dim"],
                      failure_level=config["failure_level"],
                      tol=config["tol"])
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
                      tol=config["tol"])
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
                      horizon=config["horizon"])
    elif config["algo_name"] == 'OFULogPlus':
        algo = OFULogPlus(param_norm_ub=config["param_norm_ub"],
                          arm_norm_ub=config["arm_norm_ub"],
                          dim=config["dim"],
                          failure_level=config["failure_level"],
                      horizon=config["horizon"],
                      tol=config["tol"])
    elif config["algo_name"] == 'OFUGLB':
        algo = OFUGLB(param_norm_ub=config["param_norm_ub"],
                      arm_norm_ub=config["arm_norm_ub"],
                      dim=config["dim"],
                      failure_level=config["failure_level"],
                      horizon=config["horizon"],
                      tol=config["tol"])
    # elif config["algo_name"] == 'OFUGLB-ball-e':
    #     algo = OFUGLBballe(param_norm_ub=config["param_norm_ub"],
    #                    arm_norm_ub=config["arm_norm_ub"],
    #                    dim=config["dim"],
    #                    failure_level=config["failure_level"],
    #                    horizon=config["horizon"])
    elif config["algo_name"] == 'OFUGLB-e':
        algo = OFUGLBe(param_norm_ub=config["param_norm_ub"],
                       arm_norm_ub=config["arm_norm_ub"],
                       dim=config["dim"],
                       failure_level=config["failure_level"],
                      horizon=config["horizon"],
                      tol=config["tol"])
    elif config["algo_name"] == 'RS-GLinCB':
        algo = RS_GLinCB(param_norm_ub=config["param_norm_ub"],
                       arm_norm_ub=config["arm_norm_ub"],
                       dim=config["dim"],
                       failure_level=config["failure_level"],
                      horizon=config["horizon"],
                      tol=config["tol"])
    elif config["algo_name"] == 'EMK':
        algo = EMK(param_norm_ub=config["param_norm_ub"],
                         arm_norm_ub=config["arm_norm_ub"],
                         dim=config["dim"],
                         failure_level=config["failure_level"],
                         horizon=config["horizon"],
                         tol=config["tol"])
    # elif config["algo_name"] == 'EMKKj':
    #     algo = EMKKj(param_norm_ub=config["param_norm_ub"],
    #                      arm_norm_ub=config["arm_norm_ub"],
    #                      dim=config["dim"],
    #                      failure_level=config["failure_level"],
    #                      horizon=config["horizon"],
    #                      tol=config["tol"])
    # elif config["algo_name"] == 'EMKKj2':
    #     algo = EMKKj2(param_norm_ub=config["param_norm_ub"],
    #                      arm_norm_ub=config["arm_norm_ub"],
    #                      dim=config["dim"],
    #                      failure_level=config["failure_level"],
    #                      horizon=config["horizon"],
    #                      tol=config["tol"])
    elif config["algo_name"] == 'EVILL':
        algo = EVILL(param_norm_ub=config["param_norm_ub"],
                         arm_norm_ub=config["arm_norm_ub"],
                         dim=config["dim"],
                         failure_level=config["failure_level"],
                         horizon=config["horizon"],
                         tol=config["tol"])
    if algo is None:
        raise ValueError("Oops. The algorithm {} is not implemented. You must choose within ({})".format(
            config["algo_name"], ''.join(['\'' + elem + '\'' + ', ' for elem in ALGOS])))
    return algo
