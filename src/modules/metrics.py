import numpy as np

from scipy.stats import spearmanr


def evaluate_race_predictions(y_true: list[int], y_pred: list[int], driver_ids: list[str]) -> dict[str, float]:
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
