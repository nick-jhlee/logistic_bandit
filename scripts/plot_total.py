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
  -pn [PN]    Parameter norm (default: 3.0)
"""

import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import numpy as np
import os
import json
import seaborn as sns

# parser
parser = argparse.ArgumentParser(description='Plot regret curves, by default for dimension=2 and parameter norm=1',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', type=int, nargs='?', default=2, help='Dimension')
parser.add_argument('-hz', type=int, nargs='?', default=2000, help='Horizon length')
parser.add_argument('-pn', type=float, nargs='+', default=3.0, help='norm of the unknown parameter')
parser.add_argument('-ast', type=str, nargs='?', default='tv_discrete',
                    help='Arm set type. Must be either fixed_discrete, tv_discrete or ball')
parser.add_argument('-Nconfidence', type=int, nargs='?', default=1000,
                    help='Number of discretizations (per axis) for confidence set plot')

args = parser.parse_args()

d = args.d
H = args.hz
arm_set_type = args.ast
N = args.Nconfidence
pn_list = args.pn

# path to logs/
# logs_dir = f'logs/{arm_set_type}_h_{H}'
# here = os.path.dirname(os.path.abspath(__file__))
regret_dir, plot_dir = "logs/regret", "logs/plot"
# logs-24-1124b

print(r"Plotting regret curves and confidence sets for $d=${}, $H=${} and arm_set_type={}".format(d, H, arm_set_type))

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

clrs = sns.color_palette("husl", 4)

fig, axes = plt.subplots(3, 4, figsize=(30, 20))
plt.rcParams.update({
    # "text.usetex": True,
    # "font.family": "sans-serif",
    # "font.sans-serif": ["Helvetica"],
    "font.size": 24})
tick_font_size = 24

cols = [fr"$S = {pn+1}$" for pn in pn_list]
rows = ['Regret plots', 'Magnified regret plots', 'Confidence sets']

for ax, col in zip(axes[0], cols):
    ax.set_title(col)

for ax, row in zip(axes[:, 0], rows):
    ax.set_ylabel(row, size='large')

lines, labels = [], []
for row in range(3):
    # plot regrets first
    for plt_idx, pn in enumerate(pn_list):
        S = pn + 1
        if row != 2:
            # accumulate results
            res_dict_mean = dict()
            res_dict_std = dict()
            for log_path in os.listdir(regret_dir):
                with open(os.path.join(regret_dir, log_path), 'r', encoding="utf8") as data:
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
                for i, algorithm in enumerate(res_dict_mean.keys()):
                    regret = res_dict_mean[algorithm]
                    std = res_dict_std[algorithm]
                    tmp, = axes[row, plt_idx].plot(regret, label=alg_dict[algorithm], color=color_dict[algorithm], alpha=alpha_dict[algorithm])
                    axes[row, plt_idx].fill_between(range(len(regret)), regret - std, regret + std, alpha=0.3,
                                                    color=color_dict[algorithm])
                    print(f'{algorithm:20s}: {regret[-1]:10g} ±{std[-1]:.2g}')
                    if plt_idx == 0 and row == 0:
                        lines.append(tmp)
                        labels.append(tmp.get_label())
            axes[row, plt_idx].tick_params(axis='both', which='major', labelsize=tick_font_size)
            axes[row, plt_idx].tick_params(axis='both', which='minor', labelsize=tick_font_size)
            ## hard coding y-axis limits
            if arm_set_type == "tv_discrete":
                if row == 0:
                    if S == 4.0:
                        axes[row, plt_idx].set_ylim([0, 500])
                        rect = Rectangle((0, 0), 10050, 35, linestyle='dashed', edgecolor='black',
                                        facecolor='None', clip_on=False, linewidth=2.0)
                        axes[row, plt_idx].add_patch(rect)
                    elif S == 6.0:
                        axes[row, plt_idx].set_ylim([0, 400])
                        rect = Rectangle((0, 0), 10050, 30, linestyle='dashed', edgecolor='black',
                                        facecolor='None', clip_on=False, linewidth=2.0)
                        axes[row, plt_idx].add_patch(rect)
                    elif S == 8.0:
                        axes[row, plt_idx].set_ylim([0, 300])
                        rect = Rectangle((0, 0), 10050, 25, linestyle='dashed', edgecolor='black',
                                        facecolor='None', clip_on=False, linewidth=2.0)
                        axes[row, plt_idx].add_patch(rect)
                    elif S == 10.0:
                        axes[row, plt_idx].set_ylim([0, 200])
                        rect = Rectangle((0, 0), 10050, 20, linestyle='dashed', edgecolor='black',
                                        facecolor='None', clip_on=False, linewidth=2.0)
                        axes[row, plt_idx].add_patch(rect)
                    else:
                        pass
                else:
                    if S == 4.0:
                        axes[row, plt_idx].set_ylim([0, 35])
                    elif S == 6.0:
                        axes[row, plt_idx].set_ylim([0, 30])
                    elif S == 8.0:
                        axes[row, plt_idx].set_ylim([0, 25])
                    elif S == 10.0:
                        axes[row, plt_idx].set_ylim([0, 20])
                    else:
                        pass
            else:
                if row == 0:
                    if S == 4.0:
                        axes[row, plt_idx].set_ylim([0, 500])
                        rect = Rectangle((0, 0), 10050, 80, linestyle='dashed', edgecolor='black',
                                        facecolor='None', clip_on=False, linewidth=2.0)
                        axes[row, plt_idx].add_patch(rect)
                    elif S == 6.0:
                        axes[row, plt_idx].set_ylim([0, 500])
                        rect = Rectangle((0, 0), 10050, 70, linestyle='dashed', edgecolor='black',
                                        facecolor='None', clip_on=False, linewidth=2.0)
                        axes[row, plt_idx].add_patch(rect)
                    elif S == 8.0:
                        axes[row, plt_idx].set_ylim([0, 500])
                        rect = Rectangle((0, 0), 10050, 60, linestyle='dashed', edgecolor='black',
                                        facecolor='None', clip_on=False, linewidth=2.0)
                        axes[row, plt_idx].add_patch(rect)
                    elif S == 10.0:
                        axes[row, plt_idx].set_ylim([0, 500])
                        rect = Rectangle((0, 0), 10050, 50, linestyle='dashed', edgecolor='black',
                                        facecolor='None', clip_on=False, linewidth=2.0)
                        axes[row, plt_idx].add_patch(rect)
                    else:
                        pass
                else:
                    if S == 4.0:
                        axes[row, plt_idx].set_ylim([0, 80])
                    elif S == 6.0:
                        axes[row, plt_idx].set_ylim([0, 70])
                    elif S == 8.0:
                        axes[row, plt_idx].set_ylim([0, 60])
                    elif S == 10.0:
                        axes[row, plt_idx].set_ylim([0, 50])
                    else:
                        pass
        ## plot confidence sets
        else:
            # basic quantities
            theta_star = np.array([(S - 1) / np.sqrt(2), (S - 1) / np.sqrt(2)])
            interact_rng = np.linspace(-S - 0.5, S + 0.5, N)
            x, y = np.meshgrid(interact_rng, interact_rng)

            fnames = [f"h{H+1}d{d}a{algorithm}n{pn}t{arm_set_type}.npz" for algorithm in alg_dict.keys()]
            # # label location for contourf
            # displacements = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
            # manual_locations = [[[(0.0, 0.0)], [(0.0, 0.0)], [(0.0, 0.0)], [(0.0, 0.0)]],
            #                     [[(0.0, 0.0)], [(0.0, 0.0)], [(0.0, 0.0)], [(0.0, 0.0)]],
            #                     [[(0.0, 0.0)], [(0.0, 0.0)], [(0.0, 0.0)], [(0.0, 0.0)]],
            #                     [[(0.0, 0.0)], [(0.0, 0.0)], [(0.0, 0.0)], [(0.0, 0.0)]]]
            # with sns.axes_style("whitegrid"):
            for i, fname in enumerate(fnames):
                algorithm = list(alg_dict.keys())[i]
                if algorithm not in ["OFUGLB", "OFUGLB-e", "EMK", "OFULogPlus"]:
                    print(f"Algorithm {algorithm} not in the list of algorithms to plot. Skipping...")
                    continue
                fname = f"{plot_dir}/{fname}"
                with np.load(fname) as data:
                    z = data['z']
                    # plt.imshow(z, extent=(x.min(), x.max(), y.min(), y.max()), origin="lower", cmap="Greys", alpha=0.5)
                    CS = axes[row, plt_idx].contour(x, y, z, levels=[0, 1], colors=color_dict[algorithm])
                    # axes[row, plt_idx].clabel(CS, CS.levels[::2], inline=True, fmt=alg_names[i], fontsize=40,
                    #                           manual=manual_locations[plt_idx][i])
                    axes[row, plt_idx].contourf(x, y, z, [1 - 1e-12, 1 + 1e-12], colors=color_dict[algorithm],
                                                alpha=0.1 + i * 0.05)

                    theta_hat = data['theta_hat']
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

plt.savefig(f"logs/total_h{H}d{d}t{arm_set_type}.pdf", dpi=400, bbox_inches='tight')
plt.savefig(f"logs/total_h{H}d{d}t{arm_set_type}.png", dpi=400, bbox_inches='tight')
# plt.show()
