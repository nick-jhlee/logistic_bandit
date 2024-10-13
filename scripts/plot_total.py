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
from matplotlib.patches import Rectangle

import numpy as np
import os
import json
import seaborn as sns

from logbexp.utils.utils import dsigmoid

# parser
parser = argparse.ArgumentParser(description='Plot regret curves, by default for dimension=2 and parameter norm=1',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', type=int, nargs='?', default=2, help='Dimension')
parser.add_argument('-hz', type=int, nargs='?', default=4002, help='Horizon length')
parser.add_argument('-ast', type=str, nargs='?', default='tv_discrete',
                    help='Arm set type. Must be either fixed_discrete, tv_discrete or ball')
parser.add_argument('-S', type=int, nargs='+', help='known norm upper bound of the unknown parameter')
parser.add_argument('-Nconfidence', type=int, nargs='?', default=2000,
                    help='Number of discretizations (per axis) for confidence set plot')

args = parser.parse_args()

d = args.d
H = args.hz
arm_set_type = args.ast
N = args.Nconfidence
S_list = args.S
print(r"Plotting regret curves and confidence sets for $d=${}, $H=${} and arm_set_type={}".format(d, H, arm_set_type))

alg_dict = {"OFUGLB": "OFUGLB", "OFUGLB-e": "OFUGLB-e", "EMK": "EMK", "RS-GLinCB": "RS-GLinCB", "OFULogPlus": "OFULog+",
            "adaECOLog": "ada-OFU-ECOLog", "EVILL": "EVILL"}
color_dict = {"OFUGLB": "red", "OFUGLB-e": "orange", "EMK": "blue", "RS-GLinCB": "black", "OFULogPlus": "green",
              "adaECOLog": "purple", "EVILL": "brown"}
alpha_dict = {"OFUGLB": 1, "OFUGLB-e": 1, "EMK": 0.4, "RS-GLinCB": 0.4, "OFULogPlus": 0.4, "adaECOLog": 0.4, "EVILL": 0.4}
clrs = sns.color_palette("husl", 4)

fig, axes = plt.subplots(3, 4, figsize=(30, 20))
plt.rcParams.update({
    # "text.usetex": True,
    # "font.family": "sans-serif",
    # "font.sans-serif": ["Helvetica"],
    "font.size": 24})
tick_font_size = 24

cols = [rf"$S={S}$" for S in S_list]
rows = ['Regret plots', 'Magnified regret plots', 'Confidence sets']

for ax, col in zip(axes[0], cols):
    ax.set_title(col)

for ax, row in zip(axes[:, 0], rows):
    ax.set_ylabel(row, size='large')

lines, labels = [], []
for row in range(3):
    # plot regrets first
    for plt_idx, S in enumerate(S_list):
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
                    # eliminate logs with undesired S
                    if not int(log_dict["param_norm_ub"]) == S:
                        continue

                    # print(algo)

                    res_dict_mean[algo] = np.array(log_dict["mean_cum_regret"])
                    res_dict_std[algo] = np.array(log_dict["std_cum_regret"])

            if len(res_dict_mean) == 0 or len(res_dict_std) == 0:
                raise ValueError("No logs found for the given parameters: d={}, H={}, S={}, and arm_set_type={}".format(d, H, S, arm_set_type))

            # plotting
            with sns.axes_style("whitegrid"):
                tmp = None
                # for algorithm in ["OFUGLB", "EMK"]:
                for algorithm in ["OFUGLB", "OFUGLB-e", "EMK", "RS-GLinCB", "OFULogPlus", "adaECOLog", "EVILL"]:
                    alg_name = alg_dict[algorithm]
                    regret = res_dict_mean[algorithm]
                    std = res_dict_std[algorithm]
                    tmp, = axes[row, plt_idx].plot(regret, label=alg_name, color=color_dict[algorithm], alpha=alpha_dict[algorithm])
                    axes[row, plt_idx].fill_between(range(len(regret)), regret - std, regret + std, alpha=0.3,
                                                    color=color_dict[algorithm])
                    if plt_idx == 0 and row == 0:
                        lines.append(tmp)
                        labels.append(tmp.get_label())
            axes[row, plt_idx].tick_params(axis='both', which='major', labelsize=tick_font_size)
            axes[row, plt_idx].tick_params(axis='both', which='minor', labelsize=tick_font_size)

            ## hard coding y-axis limits
            if row == 0:
                if S == 4:
                    axes[row, plt_idx].set_ylim([0, 160])
                    rect = Rectangle((0, 0), 4050, 35, linestyle='dashed', edgecolor='black',
                                     facecolor='None', clip_on=False, linewidth=2.0)
                    axes[row, plt_idx].add_patch(rect)
                elif S == 6:
                    axes[row, plt_idx].set_ylim([0, 200])
                    rect = Rectangle((0, 0), 4050, 25, linestyle='dashed', edgecolor='black',
                                     facecolor='None', clip_on=False, linewidth=2.0)
                    axes[row, plt_idx].add_patch(rect)
                elif S == 8:
                    axes[row, plt_idx].set_ylim([0, 500])
                    rect = Rectangle((0, 0), 4050, 20, linestyle='dashed', edgecolor='black',
                                     facecolor='None', clip_on=False, linewidth=2.0)
                    axes[row, plt_idx].add_patch(rect)
                elif S == 10:
                    axes[row, plt_idx].set_ylim([0, 1000])
                    rect = Rectangle((0, 0), 4050, 15, linestyle='dashed', edgecolor='black',
                                     facecolor='None', clip_on=False, linewidth=2.0)
                    axes[row, plt_idx].add_patch(rect)
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
            # # label location for contourf
            # displacements = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
            # manual_locations = [[[(0.0, 0.0)], [(0.0, 0.0)], [(0.0, 0.0)], [(0.0, 0.0)]],
            #                     [[(0.0, 0.0)], [(0.0, 0.0)], [(0.0, 0.0)], [(0.0, 0.0)]],
            #                     [[(0.0, 0.0)], [(0.0, 0.0)], [(0.0, 0.0)], [(0.0, 0.0)]],
            #                     [[(0.0, 0.0)], [(0.0, 0.0)], [(0.0, 0.0)], [(0.0, 0.0)]]]
            # with sns.axes_style("whitegrid"):
            i = 0
            for fname in fnames:
                alg_name = fname.split(".")[0]
                fname = f"S={S}/{arm_set_type}/{fname}"
                with np.load(fname) as data:
                    z = data['z']
                    # plt.imshow(z, extent=(x.min(), x.max(), y.min(), y.max()), origin="lower", cmap="Greys", alpha=0.5)
                    CS = axes[row, plt_idx].contour(x, y, z, levels=[0, 1], colors=color_dict[alg_name])
                    # axes[row, plt_idx].clabel(CS, CS.levels[::2], inline=True, fmt=alg_names[i], fontsize=40,
                    #                           manual=manual_locations[plt_idx][i])
                    axes[row, plt_idx].contourf(x, y, z, [1 - 1e-12, 1 + 1e-12], colors=color_dict[alg_name],
                                                alpha=0.1 + i * 0.05)

                    theta_hat = data['theta_hat']
                    i += 1
                    # axes[row, plt_idx].scatter(theta_hat[0], theta_hat[1], color=colors[i])
                    # axes[row, plt_idx].annotate(r" $\widehat{\theta}$", theta_hat + displacements[i], color=colors[i],
                    #                             fontsize=30)
            axes[row, plt_idx].scatter(theta_star[0], theta_star[1], color="black", marker='*', s=170)
            axes[row, plt_idx].annotate(r" $\theta_{\star}$", theta_star, color='black', fontsize=20)
            # axes[row, plt_idx].annotate(r" $\theta_{\star}$", theta_star + displacements[-1], color='black',
            #                             fontsize=20)

            z_ = (np.linalg.norm(np.array([x, y]), axis=0) <= S).astype(int)
            axes[row, plt_idx].contourf(x, y, z_, levels=[1 - 1e-12, 1 + 1e-12], alpha=0.1)

            if S == 4 or S == 6:
                axes[row, plt_idx].set_xlim([0, S + 0.1])
                axes[row, plt_idx].set_ylim([0, S + 0.1])
            elif S == 8:
                axes[row, plt_idx].set_xlim([-1, S + 0.1])
                axes[row, plt_idx].set_ylim([-1, S + 0.1])
            else:
                axes[row, plt_idx].set_xlim([-5, S + 0.1])
                axes[row, plt_idx].set_ylim([-5, S + 0.1])

            axes[row, plt_idx].tick_params(axis='both', which='major', labelsize=tick_font_size)
            axes[row, plt_idx].tick_params(axis='both', which='minor', labelsize=tick_font_size)

leg = fig.legend(lines, labels, loc='upper center', ncol=min(10, len(alg_dict)), prop={'size': 32},
           bbox_to_anchor=(0.5, 1.062))

# https://stackoverflow.com/questions/9706845/increase-the-linewidth-of-the-legend-lines-in-matplotlib
# set the linewidth of each legend object
for legobj in leg.legend_handles:
    legobj.set_linewidth(5.0)

fig.tight_layout()

plt.savefig(f"H={H}.pdf", dpi=400, bbox_inches='tight')
plt.savefig(f"H={H}.png", dpi=400, bbox_inches='tight')
plt.show()
