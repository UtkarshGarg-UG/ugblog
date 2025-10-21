import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Set up the figure with a clean, modern style
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
fig.suptitle('The Precision Paradox: How My Flawed Metric Punished Smart Behavior',
             fontsize=16, fontweight='bold', y=0.98)

# Colors
color_correct = '#10b981'  # Green
color_penalty = '#ef4444'  # Red
color_neutral = '#94a3b8'  # Gray

# LEFT CHART: My Initial (Wrong) Expectation
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')

# Title
ax1.text(5, 9.2, 'My Initial (Wrong) Expectation',
         ha='center', fontsize=14, fontweight='bold')
ax1.text(5, 8.7, 'Precision = 1.0 ✓',
         ha='center', fontsize=13, color=color_correct, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#d1fae5', edgecolor=color_correct, linewidth=2))

# Document representation
doc_y = 6.5
sections = [
    ("1. Introduction", color_neutral),
    ("2. Methods", color_penalty),  # To be deleted
    ("3. Results", color_neutral),
    ("4. Conclusion", color_neutral)
]

ax1.text(5, doc_y + 1, 'Original Document:', ha='center', fontsize=11, fontweight='bold')
for i, (section, color) in enumerate(sections):
    y_pos = doc_y - i * 0.7
    if "Methods" in section:
        # Show deletion with strikethrough
        ax1.text(5, y_pos, section, ha='center', fontsize=10,
                style='italic', color=color, alpha=0.5,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee2e2',
                         edgecolor=color, linewidth=1.5, linestyle='--'))
        ax1.plot([2.5, 7.5], [y_pos, y_pos], 'r-', linewidth=2, alpha=0.7)
    else:
        ax1.text(5, y_pos, section, ha='center', fontsize=10, color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8fafc',
                         edgecolor=color_neutral, linewidth=1))

# Expected changes
ax1.text(5, 2.5, 'Expected Changes:', ha='center', fontsize=11, fontweight='bold')
ax1.text(5, 2.0, '✓ Delete Section 2', ha='center', fontsize=10,
        color=color_correct, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#d1fae5',
                 edgecolor=color_correct, linewidth=1.5))
ax1.text(5, 1.4, "That's it. No other changes.", ha='center', fontsize=9,
        style='italic', color=color_neutral)

ax1.text(5, 0.5, 'Total: 1 change\nPrecision = 1.0', ha='center', fontsize=10,
        fontweight='bold', color=color_correct,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecfdf5',
                 edgecolor=color_correct, linewidth=2))

# RIGHT CHART: What Agent Actually Did
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')

# Title
ax2.text(5, 9.2, 'What Agent Actually Did',
         ha='center', fontsize=14, fontweight='bold')
ax2.text(5, 8.7, 'My Flawed Metric: Precision = 0.0 ✗',
         ha='center', fontsize=13, color=color_penalty, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#fee2e2', edgecolor=color_penalty, linewidth=2))

# Document after changes
ax2.text(5, doc_y + 1, 'After Smart Edits:', ha='center', fontsize=11, fontweight='bold')

result_sections = [
    ("1. Introduction → refs updated", True),
    ("2. Methods", False),  # Deleted
    ("3→2. Results (renumbered)", True),
    ("4→3. Conclusion (renumbered)", True)
]

displayed_idx = 0
for i, (section, show) in enumerate(result_sections):
    if show:
        y_pos = doc_y - displayed_idx * 0.7
        is_changed = "→" in section or "updated" in section
        if is_changed:
            ax2.text(5, y_pos, section, ha='center', fontsize=9,
                    color=color_penalty, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee2e2',
                             edgecolor=color_penalty, linewidth=1.5))
        else:
            ax2.text(5, y_pos, section, ha='center', fontsize=10, color=color_neutral,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8fafc',
                             edgecolor=color_neutral, linewidth=1))
        displayed_idx += 1

# Plus TOC update
ax2.text(5, 3.8, '+ TOC updated', ha='center', fontsize=9,
        color=color_penalty, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee2e2',
                 edgecolor=color_penalty, linewidth=1.5))

# What flawed metric saw
ax2.text(5, 2.8, 'My Flawed Metric Counted:', ha='center', fontsize=11, fontweight='bold')

changes = [
    "✓ Delete Section 2 (required)",
    "✗ Update TOC (penalty)",
    "✗ Renumber 3→2 (penalty)",
    "✗ Renumber 4→3 (penalty)",
    "✗ Fix cross-ref (penalty)"
]

y_start = 2.3
for i, change in enumerate(changes):
    color = color_correct if "required" in change else color_penalty
    bg_color = '#d1fae5' if "required" in change else '#fee2e2'
    ax2.text(5, y_start - i * 0.35, change, ha='center', fontsize=8.5,
            color=color, fontweight='bold' if "✗" in change else 'normal',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=bg_color,
                     edgecolor=color, linewidth=1))

ax2.text(5, 0.5, 'Total: 4 "unnecessary" changes\nPrecision = 0.0',
        ha='center', fontsize=10, fontweight='bold', color=color_penalty,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#fef2f2',
                 edgecolor=color_penalty, linewidth=2))

plt.tight_layout()
plt.savefig('/Users/utkarshgarg/Documents/Code/my-blog/static/llm-judge/precision_paradox.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("Chart saved to: /Users/utkarshgarg/Documents/Code/my-blog/static/llm-judge/precision_paradox.png")
