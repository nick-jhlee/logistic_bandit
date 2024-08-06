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
parser.add_argument('-Nconfidence', type=int, nargs='?', default=1000,
                    help='Number of discretizations (per axis) for confidence set plot')

args = parser.parse_args()

d = args.d
H = args.hz
arm_set_type = args.ast
N = args.Nconfidence
print(r"Plotting regret curves and confidence sets for $d=${}, $H=${} and arm_set_type={}".format(d, H, arm_set_type))

alg_dict = {"OFUGLB": "OFUGLB", "OFUGLB-e": "OFUGLB-e", "EMK": "EMK", "RS-GLinCB": "RS-GLinCB", "OFULogPlus": "OFULog+",
            "adaECOLog": "ada-OFU-ECOLog"}
alphas = [1, 1, 0.4, 0.4, 0.4, 0.4]
colors = ['red', 'orange', 'blue', 'black', 'green', 'purple']
clrs = sns.color_palette("husl", 4)

fig, axes = plt.subplots(3, 4, figsize=(30, 15))
plt.rcParams.update({
    # "text.usetex": True,
    # "font.family": "sans-serif",
    # "font.sans-serif": ["Helvetica"],
    "font.size": 17})

lines, labels = [], []
for row in range(3):
    # plot regrets first
    for plt_idx, S in enumerate([4, 6, 8, 10]):
        if row != 2:
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
                    # eliminate logs with undesired param norm
                    if not int(log_dict["norm_theta_star"]) == S - 1:
                        continue

                    # print(algo)

                    res_dict_mean[algo] = np.array(log_dict["mean_cum_regret"])
                    res_dict_std[algo] = np.array(log_dict["std_cum_regret"])

            if len(res_dict_mean) == 0 or len(res_dict_std) == 0:
                raise ValueError("No logs found for the given parameters.")

            # plotting
            with sns.axes_style("whitegrid"):
                tmp = None
                for i, algorithm in enumerate(["OFUGLB", "OFUGLB-e", "RS-GLinCB", "OFULogPlus", "EMK", "adaECOLog"]):
                    alg_name = alg_dict[algorithm]
                    regret = res_dict_mean[algorithm]
                    std = res_dict_std[algorithm]
                    tmp, = axes[row, plt_idx].plot(regret, label=alg_name, color=colors[i], alpha=alphas[i])
                    axes[row, plt_idx].fill_between(range(len(regret)), regret - std, regret + std, alpha=0.3,
                                                    color=colors[i])
                    if plt_idx == 0 and row == 0:
                        lines.append(tmp)
                        labels.append(tmp.get_label())
            axes[row, plt_idx].tick_params(axis='both', which='major', labelsize=15)
            axes[row, plt_idx].tick_params(axis='both', which='minor', labelsize=15)
            ## hard coding y-axis limits
            if row == 0:
                if S == 4:
                    axes[row, plt_idx].set_ylim([0, 160])
                elif S == 6:
                    axes[row, plt_idx].set_ylim([0, 200])
                elif S == 8:
                    axes[row, plt_idx].set_ylim([0, 500])
                elif S == 10:
                    axes[row, plt_idx].set_ylim([0, 1000])
                else:
                    pass
            else:
                if S == 4:
                    axes[row, plt_idx].set_ylim([0, 31])
                elif S == 6:
                    axes[row, plt_idx].set_ylim([0, 23])
                elif S == 8:
                    axes[row, plt_idx].set_ylim([0, 17])
                elif S == 10:
                    axes[row, plt_idx].set_ylim([0, 11])
                else:
                    pass
        ## plot confidence sets
        else:
            # basic quantities
            theta_star = np.array([(S - 1) / np.sqrt(2), (S - 1) / np.sqrt(2)])
            interact_rng = np.linspace(-S - 0.5, S + 0.5, N)
            x, y = np.meshgrid(interact_rng, interact_rng)

            fnames = ["OFUGLB.npz", "OFUGLB-e.npz", "EMK.npz", "OFULogPlus.npz"]
            alg_names = ["OFUGLB", "OFUGLB-e", "EMK", "OFULog+"]
            colors = ['red', 'orange', 'blue', 'green']

            tick_font_size = 24
            # if S == 5:
            #     displacements = [(-0.2, 0.0), (-0.1, -0.35), (0.0, 0.0)]
            #     manual_locations = [[(2.0, 1.0)], [(2.0, -2.0)]]   # label location for contourf
            # elif S == 10:
            #     displacements = [(0.0, 0), (-0.4, 0.0), (0.0, -0.2)]
            #     manual_locations = [[(6.0, 0.0)], [(6.0, 0.0)]]   # label location for contourf
            # else:
            #     print(f"For better plots, manually set displacements and contourf label locations for S={S}")
            displacements = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
            manual_locations = [[(0.0, 0.0)], [(0.0, 0.0)], [(0.0, 0.0)], [(0.0, 0.0)]]  # label location for contourf

            # plotting
            plt.figure(1, figsize=(12, 12))
            # with sns.axes_style("whitegrid"):
            for i, fname in enumerate(fnames):
                fname = f"S={S}/{arm_set_type}/{fname}"
                with np.load(fname) as data:
                    z = data['z']
                    # plt.imshow(z, extent=(x.min(), x.max(), y.min(), y.max()), origin="lower", cmap="Greys", alpha=0.5)
                    CS = plt.contour(x, y, z, levels=[0, 1], colors=colors[i])
                    axes[row, plt_idx].clabel(CS, CS.levels[::2], inline=True, fmt=alg_names[i], fontsize=40,
                                              manual=manual_locations[i])
                    axes[row, plt_idx].contourf(x, y, z, [1 - 1e-12, 1 + 1e-12], colors=colors[i], alpha=0.1 + i * 0.05)

                    theta_hat = data['theta_hat']
                    # print(np.linalg.norm(theta_hat))
                    axes[row, plt_idx].scatter(theta_hat[0], theta_hat[1], color=colors[i])
                    axes[row, plt_idx].annotate(r" $\widehat{\theta}$", theta_hat + displacements[i], color=colors[i],
                                                fontsize=35)
            axes[row, plt_idx].scatter(theta_star[0], theta_star[1], color="blue", marker='*', s=170)
            axes[row, plt_idx].annotate(r" $\theta_{\star}$", theta_star + displacements[-1], color='blue', fontsize=35)

            z_ = (np.linalg.norm(np.array([x, y]), axis=0) <= S).astype(int)
            axes[row, plt_idx].contourf(x, y, z_, levels=[1 - 1e-12, 1 + 1e-12], alpha=0.1)

            # plt.xlim([-S - 0.5, S + 0.5])
            # plt.ylim([-S - 0.5, S + 0.5])
            # plt.xlim([0.5, S + 0.1])
            # plt.ylim([0.5, S + 0.1])
            axes[row, plt_idx].tick_params(axis='both', which='major', labelsize=tick_font_size)
            axes[row, plt_idx].tick_params(axis='both', which='minor', labelsize=tick_font_size)

fig.legend(lines, labels, loc='upper center', ncol=min(10, len(alg_dict)), prop={'size': 17},
           bbox_to_anchor=(0.5, 1.065))
# plt.xlabel(r"$T$")
# plt.ylabel(r"$Reg^B(T)$")
fig.tight_layout()

plt.savefig(f"H={H}.pdf", dpi=400, bbox_inches='tight')
plt.savefig(f"H={H}.png", dpi=400, bbox_inches='tight')
plt.show(bbox_inches='tight')
