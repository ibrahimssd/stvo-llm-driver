import matplotlib.pyplot as plt
import numpy as np
import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from scipy.stats import ttest_rel
import pandas as pd


import os
import numpy as np
import matplotlib.pyplot as plt

def plot_ablation_heatmap(data):
    # Standardized capitalization matching final paper typography
    models = [
        'Legal-heBERT',
        'legal-bert-small',
        'legal-bert-base',
        'google-bert-base',
        'custom-legalbert',
        'Italian-Legal-BERT',
        'bert-base-contracts',
        'bert-base-eurlex',
        'bert-base-echr'
    ]

    configs = ['ALL\n(NSP+CLU+CLS)', 'NSP+CLS\n($-$CLU)', 'CLU+CLS\n($-$NSP)',
               'NSP+CLU\n($-$CLS)', 'CLS only', 'CLU only', 'NSP only']

    fig, ax = plt.subplots(figsize=(11, 7.5), facecolor='white')

    im = ax.imshow(data, cmap='YlGnBu', aspect='auto', vmin=0.3, vmax=0.9)

    ax.set_xticks(np.arange(len(configs)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(configs, fontsize=9, fontweight='bold')
    ax.set_yticklabels(models, fontsize=9, fontweight='bold')

    ax.spines[:].set_visible(False)

    # Matrix value annotations with precise high-contrast thresholding
    for i in range(len(models)):
        for j in range(len(configs)):
            color = "white" if data[i, j] > 0.68 else "black"
            ax.text(j, i, f'{data[i, j]:.3f}',
                    ha="center", va="center", color=color,
                    fontsize=9, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label('Macro-F$_1$ Score', rotation=270, labelpad=20,
                    fontsize=11, fontweight='bold')

    # Accent boundaries matching the LaTeX output
    rect_color = '#111111'
    for i in range(len(models)):
        ax.add_patch(plt.Rectangle((-0.5, i - 0.5), 1, 1,
                                   fill=False, edgecolor=rect_color,
                                   linewidth=2, zorder=10))

    # Separation horizontal split (Aligned between row 3 and 4)
    ax.axhline(y=3.5, color='#2c3e50', linewidth=1.5, linestyle='--', alpha=0.8, zorder=11)

    ax.set_title('Task Contribution Across Models',
                 fontsize=13, fontweight='bold', pad=20)
    ax.set_xlabel('Pre-training Objective Configuration', fontsize=11,
                  fontweight='bold', labelpad=12)
    ax.set_ylabel('Base Model Initializations', fontsize=11,
                  fontweight='bold', labelpad=12)

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/ablation_heatmap_unified.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('plots/ablation_heatmap_unified.png', dpi=300, bbox_inches='tight')
    print("✓ Unified ablation heatmap saved")


def plot_task_importance():
    tasks = ['CLS\n(Classification)', 'NSP\n(Next Sentence)', 'CLU\n(KG Clustering)']
    tasks_short = ['CLS', 'NSP', 'CLU']
    avg_drops = [0.277, 0.105, 0.056]
    std_devs  = [0.180, 0.119, 0.090]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(7.5, 6.2), facecolor='white')

    # Swapped multicolored bars for a unified academic palette
    bar_color = '#2c3e50' 
    error_color = '#7f8c8d'

    bars = ax.bar(tasks_short, avg_drops, yerr=std_devs,
                  color=bar_color, edgecolor='#111111', linewidth=1.0,
                  capsize=6, alpha=0.95, zorder=3,
                  error_kw={'linewidth': 1.2, 'elinewidth': 1.2, 'capthick': 1.2, 'ecolor': error_color})

    # Standardized mathematical annotation logic
    for bar, val, std in zip(bars, avg_drops, std_devs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + std + 0.015,
                f'{val:.3f}\n$\\pm${std:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='#222222')

    ax.set_ylabel('Avg. Macro-F$_1$ Drop When Removed', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_xlabel('Pre-training Objective', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title('Pre-training Objective Importance',
                 fontsize=13, fontweight='bold', pad=22)

    ax.text(0.5, 1.03, 'Average performance degradation across 9 base models',
            transform=ax.transAxes, ha='center', fontsize=9.5,
            style='italic', color='#555555')

    ax.set_xticks(range(len(tasks_short)))
    ax.set_xticklabels(tasks, fontsize=9, fontweight='bold')
    ax.tick_params(axis='y', labelsize=9)

    ax.set_ylim([0, 0.6])
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    props = dict(boxstyle='round,pad=0.5', facecolor='whitesmoke', edgecolor='gray', alpha=0.7)
    ax.text(0.95, 0.95, 'Higher value =\nGreater Importance',
            transform=ax.transAxes, fontsize=9, color='#333333',
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/task_importance.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('plots/task_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Optimized task importance plot saved.")
    
def run_stats_test(data):
    # ALL = col 0, CLM+CLS = col 1, CLU+CLS = col 2, CLM+CLU = col 3
    all_scores = data[:, 0]
    clm_cls_scores = data[:, 1]
    clu_cls_scores = data[:, 2]
    clm_clu_scores = data[:, 3]

    # CLU Contribution (ALL vs NSP+CLS)
    t_stat, p_val = ttest_rel(all_scores, clm_cls_scores)
    print(f"CLU Contribution (ALL vs NSP+CLS): t={t_stat:.4f}, p={p_val:.4f}")

    # NSP Contribution (ALL vs CLU+CLS)
    t_stat, p_val = ttest_rel(all_scores, clu_cls_scores)
    print(f"NSP Contribution (ALL vs CLU+CLS): t={t_stat:.4f}, p={p_val:.4f}")

    # CLS Contribution (ALL vs NSP+CLU)
    t_stat, p_val = ttest_rel(all_scores, clm_clu_scores)
    print(f"CLS Contribution (ALL vs NSP+CLU): t={t_stat:.4f}, p={p_val:.4f}")



def plot_transfer_gains(data_dict, output_name="transfer_analysis"):
    """
    Plots a professional grouped bar chart comparing performance gains 
    across two datasets.
    """
    # 1. Prepare and Sort Data
    df = pd.DataFrame(data_dict)
    # Sorting by Austrian Gain creates a clean visual slope
    df = df.sort_values(by='AT_Gain', ascending=False)
    
    # 2. Configure Figure Style
    plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    x_indices = np.arange(len(df))
    bar_width = 0.38
    
    # 3. Create Grouped Bars
    bars_at = ax.bar(x_indices - bar_width/2, df['AT_Gain'], bar_width, 
                     label='Austrian ($\Delta$)', color='#4c72b0', alpha=0.9)
    bars_ie = ax.bar(x_indices + bar_width/2, df['IE_Gain'], bar_width, 
                     label='Irish ($\Delta$)', color='#dd8452', alpha=0.9)
    
    # 4. Add Labels and Styling
    ax.set_ylabel('Performance Gain ($m$-$F_1$)', fontweight='bold')
    ax.set_xlabel('Model Architecture', fontweight='bold', labelpad=10)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(df['Model'], rotation=35, ha='right')
    ax.legend(loc='upper right', frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.axhline(0, color='black', linewidth=1.0) # Baseline 0 line
    
    # 5. Add Text Annotations on Bars
    def add_value_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:+.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3 if height > 0 else -12),
                        textcoords="offset points",
                        ha='center', va='bottom' if height > 0 else 'top',
                        fontsize=9, fontweight='bold')

    add_value_labels(bars_at)
    add_value_labels(bars_ie)
    
    # 6. Save and Finalize
    plt.tight_layout()
    plt.savefig(f"plots/{output_name}.pdf")
    plt.savefig(f"plots/{output_name}.png", dpi=300)
    print(f"Figures saved as plots/{output_name}.pdf and plots/{output_name}.png")







if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting LexGLUE comparison plot generation...")  
    data = np.array([
        # ALL    NSP+CLS  CLU+CLS  NSP+CLU  CLS     CLU     NSP
        [0.864,  0.688,   0.826,   0.336,   0.587,  0.331,  0.336],  # bert-echr
        [0.862,  0.833,   0.831,   0.457,   0.675,  0.336,  0.336],  # bert-eurlex
        [0.797,  0.845,   0.805,   0.390,   0.643,  0.331,  0.331],  # bert-contracts
        [0.813,  0.806,   0.651,   0.500,   0.537,  0.331,  0.331],  # italian-legal-bert
        [0.820,  0.684,   0.435,   0.349,   0.522,  0.336,  0.336],  # custom-legalbert
        [0.620,  0.630,   0.559,   0.422,   0.480,  0.331,  0.336],  # google-bert
        [0.569,  0.510,   0.477,   0.442,   0.183,  0.344,  0.331],  # legal-bert-base
        [0.537,  0.333,   0.346,   0.504,   0.302,  0.344,  0.331],  # legal-bert-small
        [0.350,  0.401,   0.355,   0.336,   0.435,  0.336,  0.336],  # Legal-heBERT
    ])

    # plot_lexglue_comparison()

    logging.info("Starting ablation heatmap generation...")
    plot_ablation_heatmap(data)

    logging.info("Starting task importance plot generation...")
    plot_task_importance()

    logging.info("Running statistical significance test...")
    run_stats_test(data)


    # 
    # --- Example Usage ---
    if __name__ == "__main__":
        results = {
            'Model': [
                'Legal-heBERT', 'google-bert-base', 'custom-legalbert', 'legal-bert-small',
                'bert-base-echr', 'bert-base-contracts', 'Italian-Legal-BERT', 'bert-base-eurlex', 'legal-bert-base'
            ],
            'AT_Gain': [0.041, 0.158, 0.126, 0.070, 0.123, 0.120, 0.085, 0.114, 0.075],
            'IE_Gain': [-0.037, 0.064, 0.036, 0.053, 0.082, 0.002, 0.062, 0.002, 0.060]
        }
        plot_transfer_gains(results)
    






    