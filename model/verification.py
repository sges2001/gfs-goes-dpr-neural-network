"""
Verification metrics for precipitation retrieval evaluation.

Provides both continuous metrics (RMSE, bias, Pearson/Spearman correlation)
and categorical metrics derived from a confusion matrix (POD, FAR, CSI,
frequency bias, accuracy, F1, ETS) computed at multiple precipitation thresholds.
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix
from pysteps.verification.spatialscores import fss


# ---------------------------------------------------------------------------
# Continuous metrics
# ---------------------------------------------------------------------------

def rmse(model_data, target_data):
    """Root mean squared error."""
    return np.sqrt(np.mean((model_data.flatten() - target_data.flatten()) ** 2))


def bias(model_data, target_data):
    """Mean bias (model − target)."""
    return np.mean(model_data.flatten() - target_data.flatten())


def pearson_corr(model_data, target_data):
    """Pearson correlation coefficient."""
    return np.corrcoef(model_data.flatten(), target_data.flatten())[0, 1]


def spearman_corr(model_data, target_data):
    """Spearman rank correlation coefficient."""
    return spearmanr(model_data.flatten(), target_data.flatten())[0]


# ---------------------------------------------------------------------------
# Categorical metrics (threshold-based)
# ---------------------------------------------------------------------------

def confusion_matrix_thresholds(model_data, target_data, thresholds):
    """
    Compute normalized confusion matrices at multiple precipitation thresholds.

    NaN values are excluded from all computations.

    Parameters
    ----------
    model_data : np.ndarray
    target_data : np.ndarray
    thresholds : array-like
        List of precipitation thresholds [mm/h].

    Returns
    -------
    cf : np.ndarray, shape (2, 2, len(thresholds))
        Normalized confusion matrix for each threshold.
        cf[i, j, k] = fraction of samples in cell (i,j) at threshold k.
        Rows: observed class (0=no rain, 1=rain).
        Columns: predicted class (0=no rain, 1=rain).
    """
    valid = ~np.isnan(model_data) & ~np.isnan(target_data)
    n_thr = len(thresholds)
    cf    = np.zeros((2, 2, n_thr))

    for k, thr in enumerate(thresholds):
        y_obs  = (target_data[valid].flatten() > thr).astype(int)
        y_pred = (model_data[valid].flatten()  > thr).astype(int)
        cf[:, :, k] = confusion_matrix(y_obs, y_pred, labels=[0, 1], normalize="all")

    return cf


def pod_score(cf):
    """
    Probability of Detection (hit rate): POD = TP / (TP + FN).
    Measures the fraction of observed rain events that were correctly forecast.
    """
    n = cf.shape[2]
    pod = np.zeros(n)
    for k in range(n):
        denom   = cf[1, 1, k] + cf[1, 0, k]
        pod[k]  = cf[1, 1, k] / denom if denom > 0 else np.nan
    return pod


def far_score(cf):
    """
    False Alarm Ratio: FAR = FP / (FP + TP).
    Measures the fraction of forecast rain events that were not observed.
    """
    n = cf.shape[2]
    far = np.zeros(n)
    for k in range(n):
        denom   = cf[0, 1, k] + cf[1, 1, k]
        far[k]  = cf[0, 1, k] / denom if denom > 0 else np.nan
    return far


def csi_score(cf):
    """
    Critical Success Index (threat score): CSI = TP / (TP + FP + FN).
    Combines hits, misses, and false alarms into a single skill score.
    """
    n = cf.shape[2]
    csi = np.zeros(n)
    for k in range(n):
        denom   = cf[1, 1, k] + cf[0, 1, k] + cf[1, 0, k]
        csi[k]  = cf[1, 1, k] / denom if denom > 0 else np.nan
    return csi


def frequency_bias_score(cf):
    """
    Frequency bias: BIAS = (TP + FP) / (TP + FN).
    Values > 1 indicate overforecasting; values < 1 indicate underforecasting.
    """
    n    = cf.shape[2]
    fbias = np.zeros(n)
    for k in range(n):
        denom     = cf[1, 1, k] + cf[1, 0, k]
        fbias[k]  = (cf[1, 1, k] + cf[0, 1, k]) / denom if denom > 0 else np.nan
    return fbias


def accuracy_score(cf):
    """
    Accuracy: ACC = (TP + TN) / (TP + TN + FP + FN).
    """
    n   = cf.shape[2]
    acc = np.zeros(n)
    for k in range(n):
        acc[k] = cf[1, 1, k] + cf[0, 0, k]   # already normalized
    return acc


def f1_score(cf):
    """
    F1 score: F1 = TP / (TP + (FP + FN) / 2).
    Harmonic mean of precision and recall.
    """
    n  = cf.shape[2]
    f1 = np.zeros(n)
    for k in range(n):
        denom  = cf[1, 1, k] + 0.5 * (cf[0, 1, k] + cf[1, 0, k])
        f1[k]  = cf[1, 1, k] / denom if denom > 0 else np.nan
    return f1


def ets_score(cf):
    """
    Equitable Threat Score (Gilbert skill score):
        ETS = (TP - TP_random) / (TP + FP + FN - TP_random)
    where TP_random = (TP + FN)(TP + FP) / N accounts for random hits.
    Returns NaN when the denominator is zero.
    """
    n   = cf.shape[2]
    ets = np.zeros(n)
    for k in range(n):
        N          = cf[:, :, k].sum()
        tp_random  = (cf[1, 1, k] + cf[1, 0, k]) * (cf[1, 1, k] + cf[0, 1, k]) / N
        denom      = cf[1, 1, k] + cf[0, 1, k] + cf[1, 0, k] - tp_random
        ets[k]     = (cf[1, 1, k] - tp_random) / denom if denom != 0 else np.nan
    return ets