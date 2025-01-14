"""
By @nick-jhlee

usage: plot_confidence.py [-h] [-ast [AST]] [-pn [PN]] [-Nconfidence [N]]

Plot confidence sets for all algorithms

optional arguments:
    -h, --help          show this help message and exit
    -d [D]              Dimension
    -hz [HZ]            Horizon length
    -pn [PN]            Parameter norm (default: 9.0)
    -ast [AST]          Dimension (default: tv_discrete)
    -Nconfidence [N]    Number of discretizations (per axis) for confidence set plot (default: 5000)
    -algos [ALGOS [ALGOS ...]]  Algorithms with which we will plot the confidence set (default: ['OFUGLB', 'OFUGLB-e', 'EMK' ,'OFULogPlus'])
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np

# parser
parser = argparse.ArgumentParser(description='Plot confidence sets for all algorithms',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', type=int, nargs='?', default=2, help='Dimension')
parser.add_argument('-hz', type=int, nargs='?', default=2000, help='Horizon length)')
parser.add_argument('-pn', type=float, nargs='?', default=3.0, help='Parameter norm')
parser.add_argument('-ast', type=str, nargs='?', default='tv_discrete', help='Arm set type. Must be either fixed_discrete, tv_discrete or ball')
parser.add_argument('-Nconfidence', type=int, nargs='?', default=1000, help='Number of discretizations (per axis) for confidence set plot')
parser.add_argument('-algos', type=str, nargs='+', default=['OFUGLB', 'OFUGLB-e', 'EMK' ,'OFULogPlus'], help='algorithms with which we will plot the confidence set')
args = parser.parse_args()

d = args.d
H = args.hz
pn = args.pn
S = int(pn + 1)
arm_set_type = args.ast
N = args.Nconfidence
algos = args.algos

# basic quantities
theta_star = np.array([pn / np.sqrt(2), pn / np.sqrt(2)])
# interact_rng = np.linspace(-S - 0.5, S + 0.5, N)
interact_rng = np.linspace(-2*S, 2*S, N)
x, y = np.meshgrid(interact_rng, interact_rng)

# fnames = ["adaECOLog.npz","OFULogr.npz", "OFULogPlus.npz"]
# alg_names = ["ada-OFU-ECOLog", "OFULog-r", "OFULog+"]
# colors = ['purple', 'green', 'red']
alg_dict = {"OFUGLB": "OFUGLB", "OFUGLB-e": "OFUGLB-e", "EVILL": "EVILL", "EMK": "EMK", "GLOC": "GLOC",
                "RS-GLinCB": "RS-GLinCB", "GLM-UCB": "GLM-UCB",
                "OFULogPlus": "OFULog+", "adaECOLog": "ada-OFU-ECOLog", "OFULog-r": "OFULog-r",
                "LogUCB1": "LogUCB1", "OL2M": "OL2M"}
color_dict = {"OFUGLB": "red", "OFUGLB-e": "orange", "EVILL": "magenta", "EMK": "blue", "GLOC": "yellow",
                "RS-GLinCB": "black", "GLM-UCB": "brown",
                "OFULogPlus": "green", "adaECOLog": "purple", "OFULog-r": "cyan",
                "LogUCB1": "pink", "OL2M": "grey"}
## color_dict proposed by Kwang
# {"OFUGLB": "red", "EMK": "orange", "RS-GLinCB": "black", "GLOC": "green",
#           "adaECOLog": "purple", "EVILL": "blue"}
alpha_dict = {"OFUGLB": 1, "OFUGLB-e": 1, "EVILL": 0.4, "EMK": 0.4, "GLOC": 0.4,
                "RS-GLinCB": 0.4, "GLM-UCB": 0.4,
                "OFULogPlus": 0.4, "adaECOLog": 0.4, "OFULog-r": 0.4,
                "LogUCB1": 0.4, "OL2M": 0.4}

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
manual_locations = [[(0.0, 0.0)], [(0.0, 0.0)], [(0.0, 0.0)], [(0.0, 0.0)]]   # label location for contourf


# plotting
plt.figure(1, figsize=(12,12))
# with sns.axes_style("whitegrid"):
for i, algorithm in enumerate(algos):
    if algorithm not in ["OFUGLB", "OFUGLB-e", "EMK", "OFULogPlus", "OFULog-r"]:
        print(f"Algorithm {algorithm} not in the list of algorithms to plot. Skipping...")
        continue
    fname = f"logs/plot/h{H}d{d}a{algorithm}n{pn}t{arm_set_type}.npz"
    with np.load(fname) as data:
        z = data['z']
        # plt.imshow(z, extent=(x.min(), x.max(), y.min(), y.max()), origin="lower", cmap="Greys", alpha=0.5)
        CS = plt.contour(x, y, z, levels=[0, 1], colors=color_dict[algorithm])
        plt.clabel(CS, CS.levels[::2], inline=True, fmt=alg_dict[algorithm], fontsize=40, manual=manual_locations[i])
        plt.contourf(x, y, z, [1-1e-12, 1+1e-12], colors=color_dict[algorithm], alpha=0.1+i*0.05)
        
        theta_hat = np.reshape(data['theta_hat'], (-1,))
        # print(np.linalg.norm(theta_hat))
        plt.scatter(theta_hat[0], theta_hat[1], color=color_dict[algorithm])
        plt.annotate(r" $\widehat{\theta}$", theta_hat + displacements[i], color=color_dict[algorithm], fontsize=35)
plt.scatter(theta_star[0], theta_star[1], color="blue", marker='*', s=170)
plt.annotate(r" $\theta_{\star}$", theta_star + displacements[-1], color='black', fontsize=35)

z_ = (np.linalg.norm(np.array([x, y]), axis=0) <= S).astype(int)
plt.contourf(x, y, z_, levels=[1-1e-12, 1+1e-12], alpha=0.1)

# plt.xlim([-S - 0.5, S + 0.5])
# plt.ylim([-S - 0.5, S + 0.5])
# plt.xlim([0.5, S + 0.1])
# plt.ylim([0.5, S + 0.1])
plt.tick_params(axis='both', which='major', labelsize=tick_font_size)
plt.tick_params(axis='both', which='minor', labelsize=tick_font_size)
# plt.axis('off')
plt.tight_layout()
plt.savefig(f"logs/confidence_h{H}d{d}n{pn}t{arm_set_type}.pdf", dpi=300)
plt.savefig(f"logs/confidence_h{H}d{d}n{pn}t{arm_set_type}.png", dpi=300)
plt.show()