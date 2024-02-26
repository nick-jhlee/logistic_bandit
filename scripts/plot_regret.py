"""
modified by @nick-jhlee
originally by @louisfaury

usage: plot_regret.py [-h] [-d [D]] [-hz [HZ]] [-ast [AST]] [-pn [PN]]

Plot regret curves

optional arguments:
  -h, --help  show this help message and exit
  -d [D]      Dimension (default: 2)
  -hz [HZ]    Horizon length (default: 4000)
  -ast [AST]  Dimension (default: fixed_discrete)
  -pn [PN]    Parameter norm (default: 1.0)
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
parser.add_argument('-hz', type=int, nargs='?', default=4000, help='Horizon length)')
parser.add_argument('-pn', type=float, nargs='?', default=1.0, help='Parameter norm')
parser.add_argument('-ast', type=str, nargs='?', default='fixed_discrete', help='Arm set type. Must be either fixed_discrete, tv_discrete or ball')
args = parser.parse_args()

d = args.d
H = args.hz
param_norm = args.pn
arm_set_type = args.ast
print(r"Plotting regret curves for $d=${}, $H=${}, $\lVert \theta_\star \rVert_2=${}, and arm_set_type={}".format(d, H, param_norm, arm_set_type))


# path to logs/
logs_dir = f'logs/{arm_set_type}_h_{H}'

# accumulate results
res_dict = dict()
for log_path in os.listdir(logs_dir):
    with open(os.path.join(logs_dir, log_path), 'r') as data:
        log_dict = json.load(data)
        # log_dict = json.load(open(os.path.join(logs_dir, log_path), 'r'))
        log_cum_regret = np.array(log_dict["cum_regret"])

        # eliminate logs with undesired dimension
        if not int(log_dict["dim"]) == d:
            continue
        # eliminate logs with undesired horizon length
        if not int(log_dict["horizon"]) == H:
            continue
        # eliminate logs with undesired arm set type
        if not str(log_dict["arm_set_type"]) == arm_set_type:
            continue
        # eliminate logs with undesired param norm
        if not int(log_dict["norm_theta_star"]) == param_norm:
            continue

        algo = log_dict["algo_name"]
        print(algo)
        res_dict[algo] = log_cum_regret

if len(res_dict) == 0:
    raise ValueError("No logs found for the given parameters.")

# plotting
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "font.size": 17})

alphas = [1, 0.6, 0.6]
alg_dict = {"OFULogPlus": "OFULog+", "adaECOLog": "ada-OFU-ECOLog", "OFULog-r": "OFULog-r"}
colors = ['red', 'purple', 'green']
clrs = sns.color_palette("husl", 4)

with sns.axes_style("whitegrid"):
    for i, algorithm in enumerate(["OFULogPlus", "adaECOLog", "OFULog-r"]):
        alg_name = alg_dict[algorithm]
        plt.plot(res_dict[algorithm], label=alg_name, color=colors[i], alpha=alphas[i])

plt.legend(loc='upper left', prop={'size': 19})
plt.xlabel(r"$T$")
plt.ylabel(r"$Reg^B(T)$")
plt.tight_layout()
plt.tick_params(axis='both', which='major', labelsize=15)
plt.tick_params(axis='both', which='minor', labelsize=15)
plt.savefig(f"{logs_dir}/regret_S={param_norm + 1}_h={H}.pdf", dpi=300)
plt.savefig(f"{logs_dir}/regret_S={param_norm + 1}_h={H}.png", dpi=300)
plt.show()