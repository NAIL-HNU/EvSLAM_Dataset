import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_multi_S_xi_curves(
    datafiles, 
    out_pdf=None, 
    curve_labels=None, 
    figsize=(10, 7),
    x_percent_interval=20,
    y_percent_interval=20,
    dpi=300,
    show=False,
):
    """
    Plot multiple S_xi curves from different data files using v_norm as weighting.
    Args:
        datafiles (list[str]): List of txt file paths. Each file should have 3 columns: ts, rve, v_norm.
        out_pdf (str or None): If provided, save the resulting figure as a PDF file with this path.
        curve_labels (list[str] or None): Optional custom legend labels for each curve; if None, use filenames.
        figsize (tuple): Figure size in inches (width, height).
        x_percent_interval (int): Interval (in %) for x-axis ticks.
        y_percent_interval (int): Interval (in %) for y-axis ticks.
        dpi (int): Dots per inch for saved PDF.
        show (bool): Whether to display the figure with plt.show().
    Returns:
        auc_dict (dict): Mapping from filename to its normalized AUC value.
    """
    num_curves = len(datafiles)
    colors = plt.cm.tab10(np.linspace(0, 1, num_curves))
    auc_dict = {}
    plt.figure(figsize=figsize)
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    # Configure percentage-based ticks
    x_ticks = np.arange(0, 1.01, x_percent_interval/100)
    y_ticks = np.arange(0, 105, y_percent_interval)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 100])
    # Set percentage tick labels
    ax.set_xticklabels([f"{int(x*100)}%" for x in x_ticks])
    ax.set_yticklabels([f"{int(y)}%" for y in y_ticks])

    # Font and axis label settings
    plt.xlabel(r'$\xi$ (Error threshold)', fontsize=22, weight='bold')
    plt.ylabel(r'$S_{\xi}$ (Success Rate)', fontsize=22, weight='bold')
    plt.tick_params(axis='both', labelsize=18)
    
    curve_handles = []
    legend_entries = []
    xi_values = np.linspace(0, 1, 500)
    for idx, file in enumerate(datafiles):
        # Load data (expecting 3 columns: ts, rve, v_norm)
        data = np.loadtxt(file)
        print(f"Loaded {data.shape[0]} entries from {file}")
        rve = data[:, 1]
        v_norm = data[:, 2]
        # Use v_norm as weight
        weight = v_norm
        # Compute S_xi curve (weighted success rate)
        S_xi = [100 * np.sum(weight[rve < xi]) / np.sum(weight) for xi in xi_values]
        S_xi = np.array(S_xi)
        # Calculate normalized AUC
        auc = np.trapz(S_xi, xi_values) / 100
        auc_dict[os.path.basename(file)] = auc
        # Plot curve
        h, = plt.plot(xi_values, S_xi, lw=3, color=colors[idx], label=None)
        curve_handles.append(h)
        # Curve label for legend
        if curve_labels:
            label = curve_labels[idx]
        else:
            label = os.path.splitext(os.path.basename(file))[0]
        legend_entries.append(f"{label} (AUC={auc:.3f})")
    # Reference line at 100% success rate
    plt.plot([0, 1], [100, 100], 'k--', lw=2, alpha=0.7, zorder=-1)
    plt.legend(curve_handles, legend_entries, fontsize=18, loc='lower right')
    plt.tight_layout()

    if out_pdf:
        plt.savefig(out_pdf, bbox_inches='tight', dpi=dpi, transparent=True)
        print(f"Saved PDF: {out_pdf}")
    if show:
        plt.show()
    plt.close()
    return auc_dict
