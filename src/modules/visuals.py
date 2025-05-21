import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from typing import Dict, List, Optional, Tuple

from ..utils import OUT_FOLDER


def plot_position_comparison(
    actual_positions: List[int],
    predicted_positions: List[int],
    driver_names: List[str],
    constructors: Optional[List[str]] = None,
    title: str = "Race Positions: Actual vs Predicted",
    race_info: Optional[Tuple[int, int, str]] = None
) -> Figure:
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(18, 12), dpi=100)

    data = list(zip(driver_names, actual_positions, predicted_positions))
    if constructors:
        data = list(zip(driver_names, actual_positions, predicted_positions, constructors))
        data.sort(key=lambda x: x[1])
        driver_names, actual_positions, predicted_positions, constructors = map(list, zip(*data))
    else:
        data.sort(key=lambda x: x[1])
        driver_names, actual_positions, predicted_positions = map(list, zip(*data))

    x = np.arange(len(driver_names))
    width = 0.35

    if constructors:
        unique_constructors = sorted(set(constructors))
        cmap = plt.cm.get_cmap('tab20')
        color_map = {team: cmap(i % 20) for i, team in enumerate(unique_constructors)}

        actual_colors = [color_map[team] for team in constructors]

        predicted_colors = []
        for team in constructors:
            base_color = np.array(color_map[team])
            lighter_color = 0.7 * base_color + 0.3 * np.array([1, 1, 1, 1])
            predicted_colors.append(lighter_color)
    else:
        actual_colors = ['#3498DB'] * len(driver_names)
        predicted_colors = ['#A4CCE8'] * len(driver_names)

    actual_hatch = ''
    predicted_hatch = '///'

    actual_bars = ax.bar(x - width/2, actual_positions, width, label='Actual', 
                        color=actual_colors, edgecolor='black', linewidth=0.8, hatch=actual_hatch)
    predicted_bars = ax.bar(x + width/2, predicted_positions, width, label='Predicted', 
                           color=predicted_colors, edgecolor='black', linewidth=0.8, hatch=predicted_hatch)

    differences = [pred - act for act, pred in zip(actual_positions, predicted_positions)]

    for i, diff in enumerate(differences):
        if abs(diff) >= 2:
            sign = '+' if diff > 0 else ''
            ax.annotate(f'{sign}{diff}', 
                        xy=(i, min(actual_positions[i], predicted_positions[i]) - 0.5),
                        ha='center', va='top', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='#cccccc'))

    ax.set_xlabel('Driver', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Position', fontsize=12, fontweight='bold', labelpad=10)

    ax.invert_yaxis()

    max_pos = max(max(actual_positions), max(predicted_positions))
    ax.set_ylim(max_pos + 1.5, 0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(driver_names, rotation=45, ha='right', fontsize=10)

    ax.set_yticks(np.arange(1, max_pos + 1))

    if race_info:
        season, round_num, race_name = race_info
        title = f"{race_name} (Season {season}, Round {round_num}): Position Comparison"

    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)

    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bars, positions in [(actual_bars, actual_positions), (predicted_bars, predicted_positions)]:
        for bar, pos in zip(bars, positions):
            height = bar.get_height()
            ax.annotate(f'{pos}',
                        xy=(bar.get_x() + bar.get_width()/2, pos),
                        xytext=(0, -12),
                        textcoords="offset points",
                        ha='center', va='center', 
                        fontsize=9, fontweight='bold',
                        color='white',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#555555', alpha=0.8))

    legend_params = {
        'handlelength': 1.5,
        'handletextpad': 0.5,
        'columnspacing': 1.0,
        'borderpad': 0.8,
        'labelspacing': 1.0,
        'frameon': True,
        'fontsize': 10,
    }

    from matplotlib.patches import FancyBboxPatch
    legend_box_style = {'boxstyle': 'square, pad=0.6'}

    actual_patch = Patch(facecolor='grey', edgecolor='black', label='Actual')
    predicted_patch = Patch(facecolor='lightgrey', hatch='///', edgecolor='black', label='Predicted')

    type_legend = ax.legend(
        handles=[actual_patch, predicted_patch],
        loc='upper left', 
        bbox_to_anchor=(1.01, 1),
        title="Position Type",
        title_fontsize=11,
        **legend_params
    )

    legend_width = None
    if constructors:

        max_name_len = max(len(name) for name in unique_constructors)
        legend_width = max(4.5, max_name_len * 0.2)

        constructor_handles = [Patch(facecolor=color_map[team], edgecolor='black', label=team) 
                              for team in unique_constructors]

        ax.add_artist(type_legend)

        constructor_legend = ax.legend(
            handles=constructor_handles, 
            labels=unique_constructors, 
            loc='upper left', 
            bbox_to_anchor=(1.01, 0.85),
            title="Constructors",
            title_fontsize=11,
            **legend_params
        )

        type_frame = type_legend.get_frame()
        constructor_frame = constructor_legend.get_frame()

        for frame in [type_frame, constructor_frame]:
            frame.set_linewidth(0.8)
            frame.set_edgecolor('#DDDDDD')

    plt.subplots_adjust(right=0.82)

    filepath = os.path.join(OUT_FOLDER, "forecast.png")
    os.makedirs(OUT_FOLDER, exist_ok=True)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    logging.info(f"Saved Position Comparison Chart -> {filepath}")

    plt.close(fig)
    return fig

def plot_evaluation_metrics(
    metrics: Dict[str, float], 
    title: str = "Model Performance Metrics",
    race_info: Optional[Tuple[int, int, str]] = None
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

    bars = ax.bar(labels, values, color=colors, width=0.6, 
                  edgecolor='black', linewidth=0.8, alpha=0.85)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.axhline(1.0, color='#555555', linestyle='--', alpha=0.6, linewidth=1)

    if race_info:
        season, round_num, race_name = race_info
        title = f"{race_name} (Season {season}, Round {round_num}): Model Performance"

    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Score', fontsize=12, fontweight='bold', labelpad=10)

    ax.grid(axis='y', linestyle='--', alpha=0.5)

    filepath = os.path.join(OUT_FOLDER, "eval-metrics.png")
    os.makedirs(OUT_FOLDER, exist_ok=True)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    logging.info(f"Saved Evaluation Metrics Chart -> {filepath}")

    plt.close(fig)
    return fig

def visualize_race_predictions(
    display_df: pd.DataFrame,
    metrics: Optional[Dict[str, float]] = None,
    race_info: Optional[Tuple[int, int, str]] = None
) -> None:
    os.makedirs(OUT_FOLDER, exist_ok=True)

    if 'Actual Position' in display_df.columns:
        actual_positions = display_df['Actual Position'].tolist()
        predicted_positions = display_df['Predicted Position'].tolist()
        driver_names = display_df['Driver'].tolist()

        if 'Constructor' in display_df.columns:
            constructors = display_df['Constructor'].tolist()
        else:
            constructors = None

        plot_position_comparison(
            actual_positions, 
            predicted_positions, 
            driver_names, 
            constructors,
            race_info=race_info
        )
    elif 'Predicted Position' in display_df.columns:
        logging.info("No Actual Positions Available")

    if metrics:
        plot_evaluation_metrics(
            metrics, 
            race_info=race_info
        )
