"""
Loss functions for the DPR-GPM precipitation retrieval model.

Available loss classes (to be instantiated in main_train.py):
    - MSE             : Masked mean squared error.
    - MSE_Softmax     : MSE weighted by a softmax of the target values.
    - MSE_inv_PDF     : MSE weighted by the inverse of the target PDF
                        (fitted as an exponential).
    - Quantile_loss   : Asymmetric quantile (pinball) loss.
    - MultiLoss       : Combined loss penalizing MSE, spatial variance,
                        skewness, and kurtosis mismatches.

All loss classes inherit from torch.nn.Module. The mask applied in MSE,
MSE_Softmax, and Quantile_loss (target <= 1) filters values above the
maximum normalized target value, removing out-of-range samples.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dataset as ds


# ---------------------------------------------------------------------------
# Simple masked MSE
# ---------------------------------------------------------------------------

class MSE(nn.Module):
    """
    Mean squared error computed only over normalized target values <= 1.
    Values above 1 correspond to out-of-range samples after [0,1] normalization
    and are excluded from the loss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        mask = target <= 1
        loss = (output[mask] - target[mask]) ** 2
        return torch.mean(loss)


# ---------------------------------------------------------------------------
# MSE weighted by softmax of target values
# ---------------------------------------------------------------------------

class MSE_Softmax(nn.Module):
    """
    MSE loss where each sample is weighted by the softmax of the target values,
    scaled by alpha. Higher alpha concentrates weight on large precipitation values.

    Parameters
    ----------
    alpha : float
        Scaling factor applied to the target before softmax weighting.
        Larger values give more weight to high precipitation samples.
    """
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, output, target):
        mask    = target <= 1
        weights = F.softmax(self.alpha * target[mask], dim=0)
        loss    = (output[mask] - target[mask]) ** 2
        return torch.mean(loss * weights)


# ---------------------------------------------------------------------------
# MSE weighted by the inverse of the target PDF
# ---------------------------------------------------------------------------

def get_weights(dset, max_weight=1.0e2, epsilon=0, hist_flattening_factor=1.0):
    """
    Fit an exponential curve to the training target PDF and return its
    coefficients. The loss weight for each sample is the inverse of the
    fitted density at that value, capped at max_weight.

    Parameters
    ----------
    dset : dict
        Dataset dictionary containing 'TrainDataSet' (output of ds.get_Data).
    max_weight : float
        Maximum allowed weight to avoid extreme upweighting of rare values.
    epsilon : float
        Small constant added before log to avoid log(0).
    hist_flattening_factor : float
        Divides the fitted exponential slope, redistributing weights more evenly.

    Returns
    -------
    a, b : float
        Coefficients of the fitted log-linear curve: log(density) ≈ a*x + b.
    """
    _, _, target = dset["TrainDataSet"].get_data()
    target_np    = target.numpy() if hasattr(target, "numpy") else target

    valid  = target_np[target_np <= 275].flatten()
    counts, edges = np.histogram(valid)
    edges  = 0.5 * (edges[:-1] + edges[1:])
    density = counts / counts.sum()

    log_density = np.log(density + epsilon)
    a, b = np.polyfit(edges, log_density, 1)
    a    = a / hist_flattening_factor

    return a, b


class MSE_inv_PDF(nn.Module):
    """
    MSE loss weighted by the inverse of the precipitation PDF.

    Each sample is upweighted by the inverse of the fitted exponential density
    at its (denormalized) target value, concentrating the loss on rare,
    high-precipitation events.

    Parameters
    ----------
    coef_a, coef_b : float
        Coefficients of the log-linear PDF fit (from get_weights).
    max_weight : float
        Cap on the per-sample weight.
    dset : dict
        Dataset dictionary (used for denormalization).
    weight_factor : float
        Additional multiplicative scaling applied to the weights.
    """
    def __init__(self, coef_a, coef_b, max_weight, dset, weight_factor=1):
        super().__init__()
        self.coef_a        = coef_a
        self.coef_b        = coef_b
        self.max_weight    = max_weight
        self.dset          = dset
        self.weight_factor = weight_factor

    def forward(self, output, target):
        mask   = target <= 1
        t      = target[mask]
        o      = output[mask]

        # Compute density at each (denormalized) target value
        density = torch.exp(self.coef_a * ds.desnormalizacion_y(t, self.dset) + self.coef_b)

        # Weight = inverse density, capped at max_weight
        weight              = 1.0 / density
        weight[weight > self.max_weight] = self.max_weight

        loss = (o - t) ** 2 * (weight * self.weight_factor) ** 2
        return torch.mean(loss)


# ---------------------------------------------------------------------------
# Quantile (pinball) loss
# ---------------------------------------------------------------------------

class Quantile_loss(nn.Module):
    """
    Asymmetric quantile (pinball) loss for estimating a given quantile of the
    precipitation distribution.

    For a quantile alpha:
        L = mean( max(alpha * (y - ŷ), 0) + max((1-alpha) * (ŷ - y), 0) )

    Setting alpha close to 1 penalizes underestimation heavily, allowing the
    model to estimate high quantiles (e.g. alpha=0.95 targets the 95th percentile).

    Parameters
    ----------
    alpha : float
        Target quantile in (0, 1). Default 0.999 targets the extreme upper tail.
    """
    def __init__(self, alpha=0.999):
        super().__init__()
        self.alpha = alpha

    def forward(self, output, target):
        mask    = target <= 1
        t, o    = target[mask], output[mask]
        zeros   = torch.zeros_like(t)
        under   = torch.maximum(self.alpha * (t - o), zeros)
        over    = torch.maximum((1 - self.alpha) * (o - t), zeros)
        return torch.mean(under + over)


# ---------------------------------------------------------------------------
# Multi-term combined loss
# ---------------------------------------------------------------------------

def _loss_mse(output, target, weighted=False):
    """
    Standard or precipitation-weighted MSE.

    Parameters
    ----------
    weighted : bool
        If True, weights each sample by (target + eps), giving more importance
        to high-precipitation values.
    """
    o, t = output.squeeze(), target.squeeze()
    if weighted:
        eps = 1e-6
        return torch.mean((t + eps) * (o - t) ** 2)
    return torch.mean((o - t) ** 2)


def _loss_var(output, target):
    """
    MSE between the spatial variance of output and target (image-wise).
    Penalizes differences in the spatial spread of the precipitation field.
    """
    var_o = torch.var(output, dim=(1, 2))
    var_t = torch.var(target, dim=(1, 2))
    return torch.mean((var_o - var_t) ** 2)


def _loss_skew(output, target, eps=1e-3):
    """
    MSE between the spatial skewness of output and target (image-wise).
    Penalizes differences in the asymmetry of the precipitation distribution.
    Values are clamped to [-10, 10] after standardization to avoid instability.
    """
    mean_o = torch.mean(output, dim=(1, 2))
    mean_t = torch.mean(target, dim=(1, 2))
    std_o  = torch.std(output, dim=(1, 2)) + eps
    std_t  = torch.std(target, dim=(1, 2)) + eps

    norm_o = torch.clamp((output - mean_o[:, None, None]) / std_o[:, None, None], -10, 10)
    norm_t = torch.clamp((target - mean_t[:, None, None]) / std_t[:, None, None], -10, 10)

    if torch.isnan(norm_o).any() or torch.isinf(norm_o).any():
        print("  [WARNING] NaN/Inf detected in skewness normalization. Returning 0.")
        return torch.tensor(0.0, device=output.device)

    skew_o = torch.mean(norm_o ** 3, dim=(1, 2))
    skew_t = torch.mean(norm_t ** 3, dim=(1, 2))
    return torch.mean((skew_o - skew_t) ** 2)


def _loss_kurt(output, target, eps=1e-3):
    """
    MSE between the spatial kurtosis of output and target (image-wise).
    Penalizes differences in the tail heaviness of the precipitation distribution.
    Uses the same clamping strategy as _loss_skew for numerical stability.
    """
    mean_o = torch.mean(output, dim=(1, 2))
    mean_t = torch.mean(target, dim=(1, 2))
    std_o  = torch.std(output, dim=(1, 2)) + eps
    std_t  = torch.std(target, dim=(1, 2)) + eps

    norm_o = torch.clamp((output - mean_o[:, None, None]) / std_o[:, None, None], -10, 10)
    norm_t = torch.clamp((target - mean_t[:, None, None]) / std_t[:, None, None], -10, 10)

    kurt_o = torch.mean(norm_o ** 4, dim=(1, 2))
    kurt_t = torch.mean(norm_t ** 4, dim=(1, 2))
    return torch.mean((kurt_o - kurt_t) ** 2)


class MultiLoss(nn.Module):
    """
    Combined loss that jointly penalizes MSE and spatial statistic mismatches
    (variance, skewness, kurtosis) between the predicted and target fields.

    The total loss is:
        L = λ_mse * MSE + λ_var * Var + λ_skew * Skew + λ_kurt * Kurt

    A warmup period can be set for the variance term (var_warmup_epochs), during
    which λ_var is forced to 0 to allow the model to first minimize MSE before
    introducing the structural penalty.

    Parameters
    ----------
    lambda_mse, lambda_var, lambda_skew, lambda_kurt : float
        Weights for each loss term.
    use_mask : bool
        If True, restrict the loss to samples where target <= mask_threshold.
    mask_threshold : float
        Threshold for the optional target mask (default: 1.0, the normalized maximum).
    var_warmup_epochs : int
        Number of initial epochs during which the variance term is disabled.
    """
    def __init__(self, lambda_mse=1.0, lambda_var=1.0,
                 lambda_skew=5e-7, lambda_kurt=5e-8,
                 use_mask=False, mask_threshold=1.0,
                 var_warmup_epochs=0):
        super().__init__()
        self.lambda_mse        = lambda_mse
        self.lambda_var        = lambda_var
        self.lambda_skew       = lambda_skew
        self.lambda_kurt       = lambda_kurt
        self.use_mask          = use_mask
        self.mask_threshold    = mask_threshold
        self.var_warmup_epochs = var_warmup_epochs
        self.current_epoch     = 0

    def set_epoch(self, epoch):
        """Update the current epoch (called from the training loop)."""
        self.current_epoch = epoch

    def forward(self, output, target):
        if self.use_mask:
            mask   = target <= self.mask_threshold
            output = output[mask]
            target = target[mask]

        # Disable variance term during warmup to stabilize early training
        lambda_var = 0.0 if self.current_epoch < self.var_warmup_epochs else self.lambda_var

        mse  = self.lambda_mse  * _loss_mse(output, target)
        var  = lambda_var       * _loss_var(output, target)
        skew = self.lambda_skew * _loss_skew(output, target)
        kurt = self.lambda_kurt * _loss_kurt(output, target)

        return mse + var + skew + kurt, mse, var, skew, kurt