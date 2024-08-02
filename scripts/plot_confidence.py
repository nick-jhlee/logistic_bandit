"""
By @nick-jhlee

usage: plot_confidence.py [-h] [-ast [AST]] [-pn [PN]] [-Nconfidence [N]]

Plot confidence sets for all algorithms

optional arguments:
  -h, --help          show this help message and exit
  -ast [AST]          Dimension (default: tv_discrete)
  -pn [PN]            Parameter norm (default: 9.0)
  -Nconfidence [N]    Number of discretizations (per axis) for confidence set plot (default: 5000)
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from logbexp.utils.utils import dsigmoid

# parser
parser = argparse.ArgumentParser(description='Plot confidence sets for all algorithms',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-pn', type=float, nargs='?', default=9.0, help='Parameter norm')
parser.add_argument('-ast', type=str, nargs='?', default='tv_discrete', help='Arm set type. Must be either fixed_discrete, tv_discrete or ball')
parser.add_argument('-Nconfidence', type=int, nargs='?', default=5000, help='Number of discretizations (per axis) for confidence set plot')
args = parser.parse_args()

param_norm = args.pn
S = int(param_norm + 1)
arm_set_type = args.ast
N = args.Nconfidence


# basic quantities
theta_star = np.array([(S - 1) / np.sqrt(2), (S - 1) / np.sqrt(2)])
interact_rng = np.linspace(-S - 0.5, S + 0.5, N)
x, y = np.meshgrid(interact_rng, interact_rng)

# fnames = ["adaECOLog.npz","OFULogr.npz", "OFULogPlus.npz"]
# alg_names = ["ada-OFU-ECOLog", "OFULog-r", "OFULog+"]
# colors = ['purple', 'green', 'red']
fnames = ["OFULogPlus.npz", "GLMUCBPlus.npz"]
alg_names = ["OFULog+", "OFUGLB"]
colors = ['green', 'red']

tick_font_size = 24


if S == 5:
    displacements = [(-0.2, 0.0), (-0.1, -0.35), (0.0, 0.0)]
    manual_locations = [[(2.0, 1.0)], [(2.0, -2.0)]]   # label location for contourf
elif S == 10:
    displacements = [(0.0, 0), (-0.4, 0.0), (0.0, -0.2)]
    manual_locations = [[(6.0, 0.0)], [(6.0, 0.0)]]   # label location for contourf
else:
    print(f"For better plots, manually set displacements and contourf label locations for S={S}")
    displacements = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
    manual_locations = [[(0.0, 0.0)], [(0.0, 0.0)], [(0.0, 0.0)]]   # label location for contourf


# plotting
plt.figure(1, figsize=(12,12))
# with sns.axes_style("whitegrid"):
for i, fname in enumerate(fnames):
    fname = f"S={S}/{arm_set_type}/{fname}"
    with np.load(fname) as data:
        z = data['z']
        # plt.imshow(z, extent=(x.min(), x.max(), y.min(), y.max()), origin="lower", cmap="Greys", alpha=0.5)
        CS = plt.contour(x, y, z, levels=[0, 1], colors=colors[i])
        plt.clabel(CS, CS.levels[::2], inline=True, fmt=alg_names[i], fontsize=40, manual=manual_locations[i])
        plt.contourf(x, y, z, [1-1e-12, 1+1e-12], colors=colors[i], alpha=0.1+i*0.05)
        
        theta_hat = data['theta_hat']
        # print(np.linalg.norm(theta_hat))
        plt.scatter(theta_hat[0], theta_hat[1], color=colors[i])
        plt.annotate(r" $\widehat{\theta}$", theta_hat + displacements[i], color=colors[i], fontsize=35)
plt.scatter(theta_star[0], theta_star[1], color="blue", marker='*', s=170)
plt.annotate(r" $\theta_{\star}$", theta_star + displacements[-1], color='blue', fontsize=35)

z_ = (np.linalg.norm(np.array([x, y]), axis=0) <= S).astype(int)
plt.contourf(x, y, z_, levels=[1-1e-12, 1+1e-12], alpha=0.1)

# plt.xlim([-S - 0.5, S + 0.5])
# plt.ylim([-S - 0.5, S + 0.5])
plt.xlim([0.5, S + 0.1])
plt.ylim([0.5, S + 0.1])
plt.tick_params(axis='both', which='major', labelsize=tick_font_size)
plt.tick_params(axis='both', which='minor', labelsize=tick_font_size)
# plt.axis('off')
plt.tight_layout()
plt.savefig(f"S={S}/{arm_set_type}/confidence_sets_S={S}.png", dpi=300)
plt.savefig(f"S={S}/{arm_set_type}/confidence_sets_S={S}.pdf", dpi=300)
plt.show()