"""Draw figures/robust_accuracy.png (the main figure of the README).

The values are copied from the printed outputs of the PGD evaluation cells:
  - PathMNIST: pathmnist.ipynb, cell "PGD Robustness: No Distillation vs. Distilled Models"
  - CIFAR-10:  cifar10.ipynb, cell "PGD robustness: no distillation vs. distilled models"
Both notebooks use the same attack settings: L-infinity PGD in [0, 1] pixel space,
epsilon = 8/255, step 2/255, 20 iterations, random start, 1 restart.

Usage: python figures/plot_robust_accuracy.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Test accuracy (%) for: no distillation, then distilled students at T = 1, 5, 10, 20, 50
RESULTS = {
    '(a) PathMNIST (histopathology)': {
        'Clean':      [86.21, 85.92, 86.24, 86.21, 84.16, 83.51],
        'CE-PGD':     [4.82, 9.69, 21.06, 28.54, 46.49, 65.10],
        'Margin-PGD': [0.18, 0.18, 0.03, 0.26, 0.18, 0.25],
    },
    '(b) CIFAR-10': {
        'Clean':      [84.49, 83.03, 83.99, 83.04, 82.72, 81.04],
        'CE-PGD':     [0.18, 0.00, 10.44, 24.47, 45.75, 63.03],
        'Margin-PGD': [0.00, 0.00, 0.00, 0.00, 0.00, 0.02],
    },
}
X_LABELS = ['No\ndistill.', '1', '5', '10', '20', '50']
COLORS = {'Clean': '#3d3d3a', 'CE-PGD': '#2a78d6', 'Margin-PGD': '#eb6834', 'reference': '#8a8a85'}
MARKERS = {'Clean': 'o', 'CE-PGD': 's', 'Margin-PGD': '^'}
DODGE = {'Clean': 0.0, 'CE-PGD': -0.09, 'Margin-PGD': 0.09}   # keeps overlapping markers visible
LEGEND = {'Clean': 'Clean accuracy', 'CE-PGD': 'Robust accuracy, CE-PGD',
          'Margin-PGD': 'Robust accuracy, Margin-PGD'}

plt.rcParams.update({
    'savefig.dpi': 200, 'savefig.bbox': 'tight', 'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 10.5, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'legend.frameon': False,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'axes.grid.axis': 'y',
    'grid.color': '#e6e6e3', 'lines.linewidth': 1.8, 'lines.markersize': 6.5,
})

x = np.arange(len(X_LABELS))
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), sharey=True)
for ax, (title, values) in zip(axes, RESULTS.items()):
    for key, v in values.items():
        xs = x + DODGE[key]
        # Distilled models: filled markers joined by a line; no distillation: one hollow marker
        ax.plot(xs[1:], v[1:], color=COLORS[key], marker=MARKERS[key], label=LEGEND[key], zorder=3)
        ax.plot(xs[:1], v[:1], linestyle='none', marker=MARKERS[key], color=COLORS[key],
                markerfacecolor='white', markeredgewidth=1.5, zorder=3)
    ax.axvline(0.5, color=COLORS['reference'], linewidth=0.8, linestyle=':')
    ax.set_xticks(x)
    ax.set_xticklabels(X_LABELS)
    ax.set_xlim(-0.4, len(x) - 0.6)
    ax.set_ylim(-4, 104)
    ax.set_xlabel('Distillation temperature T')
    ax.set_title(title)
axes[0].set_ylabel('Test accuracy (%)')
fig.tight_layout()

handles, labels = axes[0].get_legend_handles_labels()
hollow = Line2D([], [], linestyle='none', marker='o', color=COLORS['reference'],
                markerfacecolor='white', markeredgewidth=1.5)
fig.legend(handles + [hollow], labels + ['No distillation'], loc='lower center',
           bbox_to_anchor=(0.5, 1.0), ncol=4, handlelength=2.2, columnspacing=1.4, fontsize=9)
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'robust_accuracy.png'))
