import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100
    r2 = r2_score(y_true, y_pred)
    smape = np.mean(
        2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    ) * 100

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4),
        "sMAPE": round(smape, 4),
        "R2": round(r2, 4),
    }


def compute_metrics_per_horizon(y_true, y_pred):
    horizon = y_true.shape[1] if y_true.ndim > 1 else 1
    metrics_per_step = []
    for h in range(horizon):
        if y_true.ndim > 1:
            step_metrics = compute_metrics(y_true[:, h], y_pred[:, h])
        else:
            step_metrics = compute_metrics(y_true, y_pred)
        step_metrics["horizon_step"] = h + 1
        metrics_per_step.append(step_metrics)
    return metrics_per_step
