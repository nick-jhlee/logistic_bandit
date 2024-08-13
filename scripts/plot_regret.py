"""
modified by @nick-jhlee
originally by @louisfaury

usage: plot_regret.py [-h] [-d [D]] [-hz [HZ]] [-ast [AST]] [-pn [PN]]

Plot regret curves

optional arguments:
  -h, --help  show this help message and exit
  -d [D]      Dimension (default: 2)
  -hz [HZ]    Horizon length (default: 4000)
  -ast [AST]  Dimension (default: tv_discrete)
  -pn [PN]    Parameter norm (default: 10.0)
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import seaborn as sns

from logbexp.utils.utils import dsigmoid

# parser
parser = argparse.ArgumentParser(description='Plot regret curves, by default for dimension=2 and parameter norm=1',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', type=int, nargs='?', default=2, help='Dimension')
parser.add_argument('-hz', type=int, nargs='?', default=4002, help='Horizon length)')
parser.add_argument('-ast', type=str, nargs='?', default='tv_discrete',
                    help='Arm set type. Must be either fixed_discrete, tv_discrete or ball')
parser.add_argument('-S', type=int, nargs='+', help='known norm upper bound of the unknown parameter')
args = parser.parse_args()

d = args.d
H = args.hz
S_list = args.S
arm_set_type = args.ast

for S in S_list:
    plt.figure(S)
    print(r"Plotting regret curves for $d=${}, $H=${}, $S=${}, and arm_set_type={}".format(d, H, S, arm_set_type))

    # path to logs/
    # logs_dir = f'logs/{arm_set_type}_h_{H}'
    logs_dir = 'logs/'

    # accumulate results
    res_dict_mean = dict()
    res_dict_std = dict()
    for log_path in os.listdir(logs_dir):
        with open(os.path.join(logs_dir, log_path), 'r', encoding="utf8") as data:
            log_dict = json.load(data)
            algo = log_dict["algo_name"]

            # log_dict = json.load(open(os.path.join(logs_dir, log_path), 'r'))
            if "mean_cum_regret" not in log_dict.keys() or "std_cum_regret" not in log_dict.keys():
                continue
            # eliminate logs with undesired dimension
            if not int(log_dict["dim"]) == d:
                continue
            # eliminate logs with undesired horizon length
            if not int(log_dict["horizon"]) == H:
                continue
            # eliminate logs with undesired arm set type
            if not str(log_dict["arm_set_type"]) == arm_set_type:
                continue
            # eliminate logs with undesired S
            if not int(log_dict["param_norm_ub"]) == S:
                continue

            res_dict_mean[algo] = np.array(log_dict["mean_cum_regret"])
            res_dict_std[algo] = np.array(log_dict["std_cum_regret"])

    if len(res_dict_mean) == 0 or len(res_dict_std) == 0:
        raise ValueError(f"No logs found for d={d}, H={H}, S={S}, and arm_set_type={arm_set_type}")

    # plotting
    plt.rcParams.update({
        # "text.usetex": True,
        # "font.family": "sans-serif",
        # "font.sans-serif": ["Helvetica"],
        "font.size": 17})

    alg_dict = {"OFUGLB": "OFUGLB", "OFUGLB-e": "OFUGLB-e", "EMK": "EMK", "RS-GLinCB": "RS-GLinCB", "OFULogPlus": "OFULog+",
                "adaECOLog": "ada-OFU-ECOLog"}
    color_dict = {"OFUGLB": "red", "OFUGLB-e": "orange", "EMK": "blue", "RS-GLinCB": "black", "OFULogPlus": "green",
                "adaECOLog": "purple"}
    alpha_dict = {"OFUGLB": 1, "OFUGLB-e": 1, "EMK": 0.4, "RS-GLinCB": 0.4, "OFULogPlus": 0.4, "adaECOLog": 0.4}

    clrs = sns.color_palette("husl", 4)
    with sns.axes_style("whitegrid"):
        for i, algorithm in enumerate(["OFUGLB", "EMK"]):
        # for i, algorithm in enumerate(["OFUGLB", "OFUGLB-e", "RS-GLinCB", "OFULogPlus", "EMK", "adaECOLog"]):
            alg_name = alg_dict[algorithm]
            regret = res_dict_mean[algorithm]
            std = res_dict_std[algorithm]
            plt.plot(regret, label=alg_name, color=color_dict[alg_name], alpha=alpha_dict[alg_name])
            plt.fill_between(range(len(regret)), regret - std, regret + std, alpha=0.3, color=color_dict[alg_name])

    plt.legend(loc='upper left', prop={'size': 19})
    # plt.xlabel(r"$T$")
    # plt.ylabel(r"$Reg^B(T)$")
    plt.tight_layout()
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    # ## hard coding y-axis limits
    # if S == 4:
    #     plt.ylim([0, 160])
    # elif S == 6:
    #     plt.ylim([0, 200])
    # elif S == 8:
    #     plt.ylim([0, 500])
    # elif S == 10:
    #     plt.ylim([0, 1000])
    # else:
    #     pass

    if not os.path.exists(f"S={S}/{arm_set_type}"):
        os.makedirs(f"S={S}/{arm_set_type}", exist_ok=True)

    plt.savefig(f"S={S}/{arm_set_type}/regret_S={S}_h={H}.pdf", dpi=300)
    plt.savefig(f"S={S}/{arm_set_type}/regret_S={S}_h={H}.png", dpi=300)
    plt.show()
