import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'nature'])

# 画一个2行3列的图 (since XJTU has 6 batches: 0-5)
fig, axs = plt.subplots(2, 3, figsize=(8, 6), dpi=150)

color_list = [
    '#74AED4',
    '#7BDFF2',
    '#FBDD85',
    '#F46F43',
    '#CF3D3E'
]
colors = plt.cm.colors.LinearSegmentedColormap.from_list(
    'custom_cmap', color_list, N=256
)

# XJTU batch indices
batches = [0, 1, 2, 3, 4, 5]

# Plot each batch
for count, batch in enumerate(batches):
    row, col = divmod(count, 3)

    root = f'../results/Ours/XJTU results/{batch}-{batch}/Experiment1/'
    title = f'XJTU batch {batch + 1}'

    try:
        pred_label = np.load(root + 'pred_label.npy')
        true_label = np.load(root + 'true_label.npy')
    except FileNotFoundError:
        print(f"Warning: Data for XJTU batch {batch} not found, skipping.")
        continue

    error = np.abs(pred_label - true_label)

    # Plot
    ax = axs[row, col]
    sc = ax.scatter(true_label, pred_label, c=error, cmap=colors, s=3, alpha=0.7, vmin=0, vmax=0.1)
    ax.plot([0.65, 1.15], [0.65, 1.15], '--', c='#ff4d4e', alpha=1, linewidth=1)

    ax.set_aspect('equal')
    ax.set_xlabel('True SOH')
    ax.set_ylabel('Prediction')

    ax.set_xticks([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1])
    ax.set_yticks([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1])

    ax.set_xlim([0.79, 1.005])
    ax.set_ylim([0.79, 1.005])
    ax.set_title(title)

# Draw colorbar on the right outside the plots
# cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=colors, norm=plt.Normalize(vmin=0, vmax=0.1)),
#                     ax=axs,
#                     location='right',
#                     label='Absolute error',
#                     shrink=0.7)

plt.tight_layout()
plt.show()
