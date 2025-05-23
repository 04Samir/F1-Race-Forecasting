import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from typing import Dict, List, Tuple

from ..utils import OUT_FOLDER


def plot_training_history(train_losses: list[float], val_losses: list[float]) -> None:
    if not train_losses:
        logging.warning('No Training History Found - Cannot Plot')
        return

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)

    epochs = range(1, len(train_losses) + 1)

    ax.plot(
        epochs, train_losses,
        color='#1F77B4', linewidth=2.5, marker='o', markersize=4,
        markerfacecolor='white', markeredgewidth=1.5,
        label='Training Loss', alpha=0.9
    )

    ax.plot(
        epochs, val_losses,
        color='#D62728', linewidth=2.5, marker='o', markersize=4,
        markerfacecolor='white', markeredgewidth=1.5,
        label='Validation Loss', alpha=0.9
    )

    ax.fill_between(epochs, train_losses, alpha=0.1, color='#1F77B4')
    ax.fill_between(epochs, val_losses, alpha=0.1, color='#D62728')

    min_train_loss = min(train_losses)
    min_train_epoch = epochs[train_losses.index(min_train_loss)]
    min_val_loss = min(val_losses)
    min_val_epoch = epochs[val_losses.index(min_val_loss)]

    ax.scatter(min_train_epoch, min_train_loss, color='#1F77B4', s=100, zorder=5)
    ax.scatter(min_val_epoch, min_val_loss, color='#D62728', s=100, zorder=5)

    yrange = max(max(train_losses), max(val_losses)) - min(min(train_losses), min(val_losses))

    ax.annotate(
        f'Best Training [Epoch {min_train_epoch}]:\n{min_train_loss:.4f}',
        xy=(min_train_epoch, min_train_loss), xytext=(min_train_epoch, min_train_loss + yrange * 0.6),
        arrowprops=dict(facecolor='#1F77B4', edgecolor='#1F77B4', shrink=0.05, width=1.5, headwidth=8, alpha=0.7),
        horizontalalignment='center', verticalalignment='bottom',
        fontsize=9, fontweight='bold', color='#1F77B4',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='#cccccc')
    )

    ax.annotate(
        f'Best Validation [Epoch {min_val_epoch}]:\n{min_val_loss:.4f}',
        xy=(min_val_epoch, min_val_loss), xytext=(min_val_epoch, min_val_loss + yrange * 0.6),
        arrowprops=dict(facecolor='#D62728', edgecolor='#D62728', shrink=0.05, width=1.5, headwidth=8, alpha=0.7),
        horizontalalignment='center',
        verticalalignment='bottom',
        fontsize=9, fontweight='bold', color='#D62728',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='#cccccc')
    )

    ax.set_title('Training & Validation Loss', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold', labelpad=10)

    ax.grid(axis='y', linestyle='--', alpha=0.5)

    legend = ax.legend(loc='upper right', frameon=True, framealpha=1.0,
                       edgecolor='gray', fancybox=True, fontsize=10)
    legend.get_frame().set_facecolor('white')

    os.makedirs(OUT_FOLDER, exist_ok=True)
    plot_path = f'{OUT_FOLDER}/learning-curve.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    plt.close(fig)


def plot_position_comparison(
    actual_positions: List[int],
    predicted_positions: List[int],
    driver_names: List[str],
    constructors: List[str],
    race_info: Tuple[int, int, str],
) -> Figure:
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(18, 12), dpi=100)

    data = list(zip(driver_names, actual_positions, predicted_positions, constructors))
    data.sort(key=lambda x: x[1])
    driver_names, actual_positions, predicted_positions, constructors = map(list, zip(*data))

    x = np.arange(len(driver_names))
    width = 0.35

    unique_constructors = sorted(set(constructors))
    cmap = plt.cm.get_cmap('tab20')
    color_map = {team: cmap(i % 20) for i, team in enumerate(unique_constructors)}

    actual_colors = [color_map[team] for team in constructors]

    predicted_colors = []
    for team in constructors:
        base_color = np.array(color_map[team], dtype=float)
        lighter_color = 0.7 * base_color + 0.3 * np.array([1, 1, 1, 1])
        predicted_colors.append(lighter_color)

    actual_hatch = ''
    predicted_hatch = '///'

    actual_bars = ax.bar(
        x - width / 2, actual_positions, width, label='Actual',
        color=actual_colors, edgecolor='black', linewidth=0.8, hatch=actual_hatch
    )
    predicted_bars = ax.bar(
        x + width / 2, predicted_positions, width, label='Predicted',
        color=predicted_colors, edgecolor='black', linewidth=0.8, hatch=predicted_hatch
    )

    differences = [pred - act for act, pred in zip(actual_positions, predicted_positions)]

    for i, diff in enumerate(differences):
        if abs(diff) >= 2:
            sign = '+' if diff > 0 else ''
            ax.annotate(
                f'{sign}{diff}',
                xy=(i, min(actual_positions[i], predicted_positions[i]) - 0.5),
                ha='center', va='top', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='#cccccc')
            )

    ax.set_xlabel('Driver', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Position', fontsize=12, fontweight='bold', labelpad=10)

    ax.invert_yaxis()

    max_pos = max(max(actual_positions), max(predicted_positions))
    ax.set_ylim(max_pos + 1.5, 0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(driver_names, rotation=45, ha='right', fontsize=10)

    ax.set_yticks(np.arange(1, max_pos + 1))

    season, round_num, race_name = race_info
    title = f'{race_name} (Season {season}, Round {round_num}): Position Comparison'

    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)

    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bars, positions in [(actual_bars, actual_positions), (predicted_bars, predicted_positions)]:
        for bar, pos in zip(bars, positions):
            height = bar.get_height()
            ax.annotate(
                f'{pos}',
                xy=(bar.get_x() + bar.get_width() / 2, pos),
                xytext=(0, -12),
                textcoords='offset points',
                ha='center', va='center',
                fontsize=9, fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#555555', alpha=0.8)
            )

    legend_params = {
        'handlelength': 1.5,
        'handletextpad': 0.5,
        'columnspacing': 1.0,
        'borderpad': 0.8,
        'labelspacing': 1.0,
        'frameon': True,
        'fontsize': 10,
    }

    actual_patch = Patch(facecolor='grey', edgecolor='black', label='Actual')
    predicted_patch = Patch(facecolor='lightgrey', hatch='///', edgecolor='black', label='Predicted')

    type_legend = ax.legend(
        handles=[actual_patch, predicted_patch],
        loc='upper left',
        bbox_to_anchor=(1.01, 1),
        title='Position Type',
        title_fontsize=11,
        **legend_params
    )

    constructor_handles = [
        Patch(facecolor=color_map[team], edgecolor='black', label=team)
        for team in unique_constructors
    ]
    ax.add_artist(type_legend)

    constructor_legend = ax.legend(
        handles=constructor_handles,
        labels=unique_constructors,
        loc='upper left',
        bbox_to_anchor=(1.01, 0.85),
        title='Constructors',
        title_fontsize=11,
        **legend_params
    )

    type_frame = type_legend.get_frame()
    constructor_frame = constructor_legend.get_frame()

    for frame in [type_frame, constructor_frame]:
        frame.set_linewidth(0.8)
        frame.set_edgecolor('#DDDDDD')

    plt.subplots_adjust(right=0.82)

    filepath = os.path.join(OUT_FOLDER, 'forecast-comparison.png')
    os.makedirs(OUT_FOLDER, exist_ok=True)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    logging.info(f'Saved Position Comparison Chart -> {filepath}')

    plt.close(fig)
    return fig


def plot_evaluation_metrics(
    metrics: Dict[str, float],
    race_info: Tuple[int, int, str],
) -> Figure:
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)

    metric_names = {
        'spearman': 'Spearman\nCorrelation',
        'accuracy': 'Exact Position\nAccuracy',
        'within_one': 'Within 1\nPosition',
        'podium_accuracy': 'Podium\nAccuracy',
        'top5_accuracy': 'Top-5\nAccuracy'
    }

    ordered_metrics = ['spearman', 'accuracy', 'within_one', 'podium_accuracy', 'top5_accuracy']
    values = [metrics.get(metric, 0) for metric in ordered_metrics]
    labels = [metric_names.get(metric, metric) for metric in ordered_metrics]

    colors = ['#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#E74C3C']

    bars = ax.bar(
        labels, values, color=colors, width=0.6,
        edgecolor='black', linewidth=0.8, alpha=0.85
    )

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f'{height:.2f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords='offset points',
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )

    ax.axhline(1.0, color='#555555', linestyle='--', alpha=0.6, linewidth=1)

    season, round_num, race_name = race_info
    title = f'{race_name} (Season {season}, Round {round_num}): Model Performance'

    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Score', fontsize=12, fontweight='bold', labelpad=10)

    ax.grid(axis='y', linestyle='--', alpha=0.5)

    filepath = os.path.join(OUT_FOLDER, 'eval-metrics.png')
    os.makedirs(OUT_FOLDER, exist_ok=True)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    logging.info(f'Saved Evaluation Metrics Chart -> {filepath}')

    plt.close(fig)
    return fig
