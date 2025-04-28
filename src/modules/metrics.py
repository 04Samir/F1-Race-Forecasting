import os

import numpy as np
import numpy.typing as npt

from scipy.stats import spearmanr

from ..utils import OUT_FOLDER


def spearman_correlation(y_true: npt.NDArray[np.int_], y_pred: npt.NDArray[np.int_]) -> float:
    if len(y_true) == 0:
        return 0.0
    
    corr, _ = spearmanr(y_true, y_pred)
    return corr if not np.isnan(corr) else 0.0  # type: ignore


def evaluate_race_predictions(y_true: list[int], y_pred: list[int], driver_ids: list[str]) -> dict[str, float]:
    from scipy.stats import spearmanr

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    spearman_corr, _ = spearmanr(y_true_arr, y_pred_arr)

    correct_positions = np.sum(y_true_arr == y_pred_arr)
    accuracy = correct_positions / len(y_true) if len(y_true) > 0 else 0.0

    within_one = np.sum(np.abs(y_true_arr - y_pred_arr) <= 1)
    within_one_pct = within_one / len(y_true) if len(y_true) > 0 else 0.0
    
    podium_accuracy = 0.0
    top5_accuracy = 0.0

    actual_podium = set(np.array(driver_ids)[np.where(y_true_arr <= 3)[0]])
    pred_podium = set(np.array(driver_ids)[np.where(y_pred_arr <= 3)[0]])

    if len(actual_podium) > 0:
        podium_accuracy = len(actual_podium.intersection(pred_podium)) / min(3, len(actual_podium))

    actual_top5 = set(np.array(driver_ids)[np.where(y_true_arr <= 5)[0]])
    pred_top5 = set(np.array(driver_ids)[np.where(y_pred_arr <= 5)[0]])

    if len(actual_top5) > 0:
        top5_accuracy = len(actual_top5.intersection(pred_top5)) / min(5, len(actual_top5))

    return {
        'spearman': spearman_corr,  # type: ignore
        'accuracy': accuracy,
        'within_one': within_one_pct,
        'podium_accuracy': podium_accuracy,
        'top5_accuracy': top5_accuracy
    }

def format_evaluation_results(metrics: dict[str, float]) -> str:
    basic_results = f"Spearman Rank Correlation: {metrics.get('spearman', 0):.4f}\n" + \
        f"Exact Position Accuracy: {metrics.get('accuracy', 0):.4f}\n" + \
        f"Positions Within 1 Place: {metrics.get('within_one', 0):.4f}"

    basic_results += f"\nPodium Prediction Accuracy: {metrics.get('podium_accuracy', 0):.4f}"
    basic_results += f"\nTop-5 Prediction Accuracy: {metrics.get('top5_accuracy', 0):.4f}"

    return basic_results


def save_metrics(session: tuple[int, int, str], *, metrics: dict[str, float], table: str) -> None:
    if not os.path.exists(OUT_FOLDER):
        os.makedirs(OUT_FOLDER)

    season, _round, name = session
    with open(OUT_FOLDER / f"metrics_{season}_{_round}.txt", "w") as file:
        file.write(f"Season: {season}, Round: {_round} - {name}\n\n")
        file.write(table + "\n\n")
        file.write(format_evaluation_results(metrics))
