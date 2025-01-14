"""
modified by @nick-jhlee & @kwangsungjun
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
import ipdb

from logbexp.utils.utils import dsigmoid

# parser
parser = argparse.ArgumentParser(description='Plot regret curves, by default for dimension=2 and parameter norm=1',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', type=int, nargs='?', default=2, help='Dimension')
parser.add_argument('-hz', type=int, nargs='?', default=2000, help='Horizon length)')
parser.add_argument('-pn', type=float, nargs='+', default=3.0, help='norm of the unknown parameter')
parser.add_argument('-ast', type=str, nargs='?', default='tv_discrete',
                    help='Arm set type. Must be either fixed_discrete, tv_discrete or ball')
args = parser.parse_args()

d = args.d
H = args.hz
pn_list = args.pn
arm_set_type = args.ast

# path to logs/
# logs_dir = f'logs/{arm_set_type}_h_{H}'
# here = os.path.dirname(os.path.abspath(__file__))
logs_dir = "logs/regret"
# logs-24-1124b

for pn in pn_list:
    S = pn + 1
    plt.figure(S)
    print(r"Plotting regret curves for $d=${}, $H=${}, $S=${}, and arm_set_type={}".format(d, H, S, arm_set_type))

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

    alg_dict = {"OFUGLB": "OFUGLB", "OFUGLB-e": "OFUGLB-e", "EVILL": "EVILL", "EMK": "EMK", "GLOC": "GLOC",
                "RS-GLinCB": "RS-GLinCB", "GLM-UCB": "GLM-UCB",
                "OFULogPlus": "OFULog+", "adaECOLog": "ada-OFU-ECOLog", "OFULog-r": "OFULog-r",
                "LogUCB1": "LogUCB1", "OL2M": "OL2M"}
    color_dict = {"OFUGLB": "red", "OFUGLB-e": "orange", "EVILL": "brown", "EMK": "blue", "GLOC": "yellow",
                  "RS-GLinCB": "black", "GLM-UCB": "magenta",
                  "OFULogPlus": "green", "adaECOLog": "purple", "OFULog-r": "cyan",
                  "LogUCB1": "pink", "OL2M": "grey"}
    ## color_dict proposed by Kwang
    # {"OFUGLB": "red", "EMK": "orange", "RS-GLinCB": "black", "GLOC": "green",
    #           "adaECOLog": "purple", "EVILL": "blue"}
    alpha_dict = {"OFUGLB": 1, "OFUGLB-e": 1, "EVILL": 0.4, "EMK": 0.4, "GLOC": 0.4,
                    "RS-GLinCB": 0.4, "GLM-UCB": 0.4,
                    "OFULogPlus": 0.4, "adaECOLog": 0.4, "OFULog-r": 0.4,
                    "LogUCB1": 0.4, "OL2M": 0.4}

    with sns.axes_style("whitegrid"):
        for i, algorithm in enumerate(res_dict_mean.keys()):
            regret = res_dict_mean[algorithm]
            std = res_dict_std[algorithm]
            plt.plot(regret, label=algorithm, color=color_dict[algorithm], alpha=alpha_dict[algorithm])
            plt.fill_between(range(len(regret)), regret - std, regret + std, alpha=0.6, color=color_dict[algorithm])
            print(f'{algorithm:20s}: {regret[-1]:10g} ±{std[-1]:.2g}')

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

    plt.savefig(f"logs/regret_h{H}d{d}n{pn}t{arm_set_type}.pdf", dpi=300)
    plt.savefig(f"logs/regret_h{H}d{d}n{pn}t{arm_set_type}.png", dpi=300)
    plt.show()

