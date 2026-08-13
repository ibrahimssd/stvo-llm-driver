import matplotlib.pyplot as plt
import numpy as np

# Data for both metrics
tasks = ['UNFAIR-ToS', 'CaseHOLD', 'ECtHR-A', 'ECtHR-B', 'LEDGAR', 'EUR-LEX']

# Macro-F1 scores
macro_published_sota = [83.0, 75.4, 64.7, 74.7, 83.1, 57.9]
macro_our_best = [86.0, 77.0, 65.7, 73.8, 82.1, 46.6]

# Micro-F1 scores
micro_published_sota = [96.0, 75.4, 71.7, 80.4, 88.3, 72.1]
micro_our_best = [96.8, 77.0, 72.7, 81.0, 88.2, 70.3]

# Professional color palette (Academic Style)
color_sota = '#4C72B0'  # Muted Blue
color_ours = '#DD8452'  # Muted Orange

def format_ax(ax, title, ylabel, ylim):
    """Applies a clean scientific style to the axis."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(title, fontsize=16, pad=20, fontweight='medium')
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel('LexGLUE Dataset', fontsize=14)
    ax.set_ylim(ylim)
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.tick_params(axis='both', which='major', labelsize=12)

def add_labels_and_deltas(ax, x, v_sota, v_ours, width, y_offset=1.5, delta_offset=5):
    """Adds text labels and delta indicators."""
    for i, (s, o) in enumerate(zip(v_sota, v_ours)):
        # Value labels
        ax.text(i - width/2, s + y_offset, f'{s:.1f}', ha='center', fontsize=10)
        ax.text(i + width/2, o + y_offset, f'{o:.1f}', ha='center', fontsize=10)
        
        # Delta calculation
        diff = o - s
        color = 'darkgreen' if diff >= 0 else 'darkred'
        prefix = '+' if diff >= 0 else ''
        
        # Draw delta text above the higher bar
        ax.text(i, max(s, o) + delta_offset, f'{prefix}{diff:.1f}', 
                ha='center', fontsize=11, color=color, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.2', alpha=0.9))

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
x = np.arange(len(tasks))
width = 0.38

# --- Macro-F1 Plot ---
format_ax(ax1, 'Macro-F$_1$ Performance Comparison', 'Macro-F$_1$ Score (%)', (40, 100))
ax1.bar(x - width/2, macro_published_sota, width, label='Published SOTA', color=color_sota, zorder=3, alpha=0.9)
ax1.bar(x + width/2, macro_our_best, width, label='Ours (StVO)', color=color_ours, zorder=3, alpha=0.9)
add_labels_and_deltas(ax1, x, macro_published_sota, macro_our_best, width, y_offset=1, delta_offset=6)

# --- Micro-F1 Plot ---
format_ax(ax2, 'Micro-F$_1$ Performance Comparison', 'Micro-F$_1$ Score (%)', (60, 110))
ax2.bar(x - width/2, micro_published_sota, width, label='Published SOTA', color=color_sota, zorder=3, alpha=0.9)
ax2.bar(x + width/2, micro_our_best, width, label='Ours (StVO)', color=color_ours, zorder=3, alpha=0.9)
add_labels_and_deltas(ax2, x, micro_published_sota, micro_our_best, width, y_offset=1, delta_offset=5)

# Final touches
for ax in [ax1, ax2]:
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=15)
    ax.legend(frameon=False, loc='upper left', fontsize=12)

plt.tight_layout()
plt.savefig('lexglue_results_scientific.pdf', bbox_inches='tight')
plt.show()