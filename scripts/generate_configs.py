"""
Automatically creates configs, stored in configs/generated_configs/
"""

import argparse
import json
import numpy as np
import os
import ipdb

# python scripts/generate_configs.py -dims 9 -pn 10 -algos GLOC OFUGLB -r 2 -hz 20 -assu 10 -fl 0.05 -ast movielens

parser = argparse.ArgumentParser(description='Automatically creates configs, stored in configs/generated_configs/',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-dims', nargs='+', type=int, help='Dimension')
parser.add_argument('-pn', nargs='+', type=float, help='Parameter norm (||theta_star||)')
parser.add_argument('-algos', type=str, nargs='+', help='Algorithms. Must be a subset of OFULog-r, OL2M, GLOC, adaECOLog, OFULogPlus, OFUGLB, OFUGLB-e, RS-GLinCB, EVILL, EMK')
parser.add_argument('-r', type=int, nargs='?', default=10, help='# of independent runs')
parser.add_argument('-hz', type=int, nargs='?', default=2000, help='Horizon length')
parser.add_argument('-ast', type=str, nargs='?', default='tv_discrete', help='Arm set type. Must be either fixed_discrete, tv_discrete, ball, or movielens')
parser.add_argument('-ass', type=int, nargs='?', default='10', help='Arm set size')
parser.add_argument('-fl', type=float, nargs='?', default=0.05, help='Failure level, must be in (0,1)')
parser.add_argument('-plotconfidence', type=str, nargs='?', default=False, help='Plot the confidence set?')
parser.add_argument('-Nconfidence', type=int, nargs='?', default=500, help='Number of discretizations (per axis) for confidence set plot')
parser.add_argument('-tol', type=float, nargs='?', default=1e-5, help='tolerance for optimization')
parser.add_argument('-seed', type=int, nargs='?', default=9817, help='random seed')
parser.add_argument('-theta_flag', type=int, nargs='?', default=None, help='1 for theta being aligned with the largest eigenvector and 2 for the opposite')
parser.add_argument('-arm_option', type=str, nargs='?', default=None, help='\'normalize\' for normalizing')
args = parser.parse_args()

# set args (no sanity check, please read the doc)
repeat = args.r
horizon = args.hz
arm_set_type = args.ast
arm_set_size = args.ass
failure_level = args.fl
dims = np.array(args.dims)
param_norm = np.array(args.pn)
algos = args.algos
plot_confidence = (args.plotconfidence in ["True", "true"])
N_confidence = args.Nconfidence

# create right config directory
here = os.path.dirname(os.path.abspath(__file__))
config_dir = os.path.join(here, 'configs', 'generated_configs')
#config_dir = 'configs/generated_configs'
if not os.path.exists(config_dir):
    os.makedirs(config_dir)
# clear existing configs
for file in os.listdir(config_dir):
    os.remove(os.path.join(config_dir, file))

# create configs
for d in dims:
    for pn in param_norm:
        pn_ub = pn + 1 # parameter upper-bound (S = ||theta_star|| + 1)
        # pn_ub = 2 * pn # parameter upper-bound (S = 2*||theta_star||)
        # pn_ub = 4 * pn # parameter upper-bound (S = 4*||theta_star||)
        if args.theta_flag is None:
            theta_star = pn / np.sqrt(d) * np.ones([d])
        else:
            sqrt_dim = int(np.sqrt(dims[0]))
            here = os.path.dirname(os.path.abspath(__file__))
            data = np.load(os.path.join(here, '..', 'data/out_pu_qi.npz'))
            Xu = data["pu"][:, :sqrt_dim]
            Xi = data["qi"][:arm_set_size, :sqrt_dim]
            tmp = []
            if args.theta_flag == 1:
                idx = -1 # pick the largest
            else:
                idx = 0 # pick the smallest
            _, U = np.linalg.eigh(Xu.T @ Xu) # small to large
            uu = U[:,idx]
            _, V = np.linalg.eigh(Xi.T @ Xi)
            vv = V[:,idx]

            theta_star = np.outer(uu,vv).ravel()
            theta_star /= np.linalg.norm(theta_star)
            theta_star *= pn

        # #- print out some information
        # Th = theta_star.reshape(sqrt_dim, sqrt_dim)
        # res = ((Xu@Th)@Xi.T).ravel()
        # res.sort()
        # print(f'total # of unique user-item arm vectors: {len(Xu)*len(Xi)}')
        # print(f'top 10:')
        # print(res[-10:].astype(float))
        # print('bottom 10:')
        # print(res[:10])

        for algo in algos:
            config_path = os.path.join(config_dir, 'h{}d{}a{}n{}t{}.json'.format(horizon, d, algo, pn, arm_set_type))
            config_dict = {"repeat": int(repeat), "horizon": int(horizon), "dim": int(d),
                           "algo_name": algo, "theta_star": theta_star.tolist(), "param_norm_ub": int(pn_ub),
                           "failure_level": float(failure_level), "arm_set_type": arm_set_type,
                           "arm_set_size": int(arm_set_size),
                           "arm_norm_ub": 1, "norm_theta_star": float(pn),
                           "plot_confidence": bool(plot_confidence), "N_confidence": int(N_confidence),
                           "tol": args.tol,
                           "seed": args.seed,
                           "theta_flag": args.theta_flag,
                           "arm_option": args.arm_option
                           }

            with open(config_path, 'w') as f:
                json.dump(config_dict, f)
