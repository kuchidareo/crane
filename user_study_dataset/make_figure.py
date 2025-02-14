import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import product

# Data arrays
labels = ["HAR", "Human", "LLM"]
strong_positive = np.array([3.7037037, 35.22633745, 34.1563786])
positive = np.array([4.36213992, 35.30864198, 37.36625514])
neutral = np.array([13.99176955, 22.1399177, 22.4691358])
negative = np.array([31.11111111, 5.8436214, 5.02057613])
strong_negative = np.array([46.83127572, 1.48148148, 0.98765432])

# Define colors and labels for each segment
colors = ["#8DF691","#b6f49a","#eeca6c","#f28e63","#f95852"]
text_colors = ['#2c3e50', '#2c3e50', '#2c3e50', '#1A1A1A', '#1A1A1A']
category_labels = ['Very high', 'High', 'Average', 'Low', 'Very low']
categories = [strong_positive, positive, neutral, negative, strong_negative]

plt_params = {
    "figsize": [(6, 4)],
    "title_fontsize": [12],
    "xlabel_fontsize": [14],
    "xtick_fontsize": [10],
    "ytick_fontsize": [14],
    "ylabel_padding": [60],
    "legend_fontsize": [11],
    "bartext_fontsize": [10]
}

def plot_stacked_bar_chart(figsize, title_fontsize, xlabel_fontsize, xtick_fontsize,
                           ytick_fontsize, ylabel_padding, legend_fontsize, bartext_fontsize):
    """Plot a horizontal stacked bar chart with annotations using the given style parameters."""
    y_pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=figsize)

    left = np.zeros(len(labels))  # starting positions for stacking
    bar_containers = []  # store bars and their starting positions for annotation

    # Plot each category segment as a stacked horizontal bar
    for cat, color, cat_label, text_color in zip(categories, colors, category_labels, text_colors):
        bars = ax.barh(y_pos, cat, left=left, color=color, label=cat_label)
        bar_containers.append((bars, left.copy(), cat, text_color))
        left += cat  # update left positions for the next segment

    # Annotate each bar segment with its percentage value
        for bars, start_left, widths, text_color in bar_containers:
            for rect, l, width in zip(bars, start_left, widths):
                if width > 0:
                    x_center = l + width / 2
                    y_center = rect.get_y() + rect.get_height() / 2

                    label_str = f'{width:.1f}'

                    if label_str in ['3.7']:
                        x_center -= 3.5
                    elif label_str in  ['1.0', '1.5']:
                        x_center += 4
                    ax.text(x_center, y_center, label_str, ha='center', va='center',
                            fontsize=bartext_fontsize, color=text_color)

    for p in ['top','right']:
        ax.spines[p].set_visible(False)

    # Customize y-axis labels
    ax.set_yticks(y_pos)
    ax.tick_params(axis='y', length=0, pad=ylabel_padding)
    ax.set_yticklabels(labels, fontsize=ytick_fontsize)
    for label in ax.get_yticklabels():
        label.set_horizontalalignment('left')
    
    # Remove x-axis ticks if desired, but keep the x-axis label:
    # ax.set_xticks([])  # If you prefer no tick numbers
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis='x', labelsize=xtick_fontsize)

    ax.invert_yaxis()  # So the first label appears on top
    ax.set_xlabel('Percentage (%)', fontsize=xlabel_fontsize)
    # ax.set_title('Stacked Bar Chart: Sentiment Responses', fontsize=title_fontsize)
    ax.legend(bbox_to_anchor=(1.0, 1.0), fontsize=legend_fontsize)
    
    plt.tight_layout()
    plt.show()


for params in product(
    plt_params["figsize"],
    plt_params["title_fontsize"],
    plt_params["xlabel_fontsize"],
    plt_params["xtick_fontsize"],
    plt_params["ytick_fontsize"],
    plt_params["ylabel_padding"],
    plt_params["legend_fontsize"],
    plt_params["bartext_fontsize"]
    ):
    plot_stacked_bar_chart(*params)