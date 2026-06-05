"""
Dataset construction and utility functions for the DPR-GPM retrieval model.

Main entry point: get_Data(), which loads GOES-16, GFS, and DPR arrays,
applies selection criteria, splits into train/val/test sets, and returns
a dictionary of PyTorch Dataset objects ready for use with DataLoader.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

# Indices with known data quality issues, excluded from all splits
_INVALID_INDICES = [
    1680, 2699, 2700, 2701, 4109, 4114, 4115, 4287, 4428, 4650, 4651, 4652,
    4710, 4711, 5796, 5891, 6881, 7863, 9704, 10407, 10576, 10577, 11648,
,11911, 11912, 11928, 11929, 13113, 13858, 14090, 14401, 14499, 14533, 14534, 14547, 14548, 14549, 14611, 14871, 18028, 18796, 18797, 18808, 18809, 18944, 19721, 20012, 20013, 20014, 20272, 20519, 20545, 20546, 20724, 21063, 22083, 22147, 22148, 22197, 22198, 22620, 22791, 22898, 22899, 22900, 22901, 23219, 23930, 23934, 23935, 26404, 26405, 26622, 26929, 27304, 27374, 27679, 27761, 28089, 28285, 29480, 29534, 30471, 30616, 31186, 32166, 32210, 32372, 32373, 34983, 37586, 37587, 37588, 37890, 37891, 39198, 39523, 39840, 39896, 39899, 40095, 40341, 40606, 41259, 41291, 41292, 41945, 42166, 42167, 42764, 42772, 43150, 43151, 43461, 43708, 43877, 43910, 43911, 44060, 44061, 44719, 44925, 45077, 45078, 45809, 46073, 46075, 46106, 46107, 46108, 46168, 46357, 46358, 46359, 46366, 47236, 47654, 47655, 48074, 48103, 48512, 48678, 49188, 49189, 49878, 50108, 50614, 50615, 50616, 51424, 51486, 52422, 52501, 52531, 52834, 52875, 52885, 52886, 53896, 53897, 54197, 54685, 55085, 55270, 55624, 55625, 57116, 57723, 57940, 57941, 58435, 58445, 58446, 58876, 58877, 58878, 59200, 59550, 59745, 59788, 60141, 60142, 60219, 60455, 60534, 60535, 60645, 60744, 61152, 61248, 61511, 61597, 61952, 62014, 62047, 62102, 62204, 62261, 62262, 62475, 62576, 62649, 62775, 62860, 62861, 62970, 62971, 63005, 63521, 63543, 64017, 64047, 64048, 64049, 64050, 64051, 64119, 64120, 64121, 64230, 64231, 64232, 64233, 64234, 64235, 64236, 64470, 64471, 64701, 64802, 64803, 64946, 64947, 65683, 66512, 66513, 66655, 66712, 66713, 66724, 67234, 67243, 67917, 67918, 68424, 68425, 68508, 68651, 68738, 69107, 69468, 69469, 69696, 70216, 70217, 70482, 70500, 70507, 70651, 70652, 71065, 71248, 71500, 71778, 72640, 72902, 73447, 73448, 73472, 73473, 73673, 74403, 74404, 74405, 75278, 75886, 75919, 76928, 76929, 77622, 77623, 77872, 78708, 78709, 78710, 78719, 79871, 80553, 81709, 81713, 81714, 82573, 82574, 82898, 82903, 82904, 82919, 83369, 83859, 83983, 83984, 84105, 84106, 85063, 85122, 85123, 85594, 85717, 85889, 85890, 85891, 86436, 88777, 88864, 88911, 89294, 89295, 89427, 89430, 89434, 89833, 89834, 90311, 90312, 90502, 91008, 91009, 91023, 91105, 91619, 92023, 92629, 92631, 92640, 92641, 93031, 93039, 93460, 93461, 93462, 93632, 93637, 93844, 93961, 94254, 94255, 94290, 94291, 94340, 94380, 94640, 95241, 95243, 95852, 95867, 95868, 95915, 95919, 95963, 96047, 96468, 96469, 96511, 96579, 96788, 97433, 97434, 97474, 97475, 97687, 97774, 97896, 98761, 98790, 99142, 99228, 99295, 99297, 99424, 99491, 99691, 99741, 99794, 100100, 100689, 101013, 101022, 101649, 101894, 102022, 102026, 102221, 102290, 102692, 103027, 103028, 103624, 103798, 104032, 104382, 104501, 104667, 104668, 104696, 105192, 105960, 106397, 106843, 107357, 107920, 108427, 108463, 108464, 108465, 108829, 108830, 109035, 109066, 109222, 109223, 109224, 109225, 109226, 109227, 109540, 109587, 109589, 109748, 109749, 109894, 109895, 110030, 110083, 110207, 110212, 110213, 110248, 110249, 110302, 110325, 110851, 111867, 112017, 112018, 112150, 112349, 112387, 112544, 112690, 113143, 113244, 113245, 113402, 113441, 113470]

# Index range without available GFS forecasts
_MISSING_FORECAST_RANGE = (16530, 16584)

# Test set index ranges in the original metadata (Jan 2021 + Jun 2021)
_TEST_RANGE_SUMMER = (193802, 200189)
_TEST_RANGE_WINTER = (224818, 230993)

# Maximum physically valid precipitation value [mm/h]
MAX_PRECIP = 275.0

# Minimum fraction of rainy DPR pixels to include an image in training
DPR_RAIN_THR = 0.15

# Minimum (negative) correlation between DPR and GOES-RRQPE to include an image
CORRELATION_THR = -0.15

# GFS forecast spatial downscaling: reshape (96,96) → (6,8,6,8) and average
GFS_DOWNSCALE_SHAPE = (6, 8, 6, 8)


def get_Data(input_sat, input_gfs, target, n_file, conf):
    """
    Build train, validation, and test PyTorch datasets from raw input arrays.

    Applies quality filters (NaN removal, blacklisted indices, DPR rain coverage,
    GOES-DPR correlation), removes the test set, splits the remainder into train
    and validation, computes normalization statistics from training data only,
    and returns a dataset dictionary.

    Parameters
    ----------
    input_sat : np.ndarray, shape (N, H, W)
        GOES-16 brightness temperature images.
    input_gfs : np.ndarray, shape (N, H_gfs, W_gfs)
        GFS forecast fields, interpolated to the GOES grid.
    target : np.ndarray, shape (N, H, W)
        DPR precipitation fields (target variable).
    n_file : np.ndarray, shape (N,)
        Index mapping each dataset sample to the original metadata row.
    conf : dict
        Training configuration dictionary (from main_train.py).

    Returns
    -------
    dict
        Dictionary containing 'TrainDataSet', 'ValDataSet', 'TestDataSet',
        normalization statistics, and dataset metadata.
    """
    print(f"Total samples before filtering: {input_sat.shape[0]}")

    # Spatially downscale GFS forecasts to match model input resolution
    n = input_gfs.shape[0]
    input_gfs = input_gfs.reshape(n, *GFS_DOWNSCALE_SHAPE).mean(axis=(2, 4))
    print(f"GFS input shape after downscaling: {input_gfs.shape}")

    dataset = {
        "nt": input_sat.shape[0],
        "nx": input_sat.shape[1],
        "ny": input_sat.shape[2],
        "Norm":      conf["Norm"],
        "TypeNorm":  conf["TypeNorm"],
        "Device":    conf["Device"],
        "BatchSize": conf["BatchSize"],
        "Shuffle":   conf["Shuffle"],
        "Transform": conf["Transform"],
    }

    train_val_indices, test_ids = _build_splits(
        input_sat, input_gfs, target, n_file,
        dataset["nx"], dataset["ny"]
    )

    # Remove indices in the missing forecast range
    missing_mask      = ((train_val_indices < _MISSING_FORECAST_RANGE[0]) |
                         (train_val_indices > _MISSING_FORECAST_RANGE[1]))
    removed           = train_val_indices[~missing_mask]
    train_val_indices = train_val_indices[missing_mask]
    if len(removed) > 0:
        print(f"Removed {len(removed)} indices in missing forecast range "
              f"[{_MISSING_FORECAST_RANGE[0]}, {_MISSING_FORECAST_RANGE[1]}].")

    train_ids, val_ids = train_test_split(
        train_val_indices,
        test_size=1 - conf["TrainRatio"],
        shuffle=False
    )
    print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

    # Convert to tensors with time axis last for indexing
    input_sat  = torch.tensor(np.moveaxis(input_sat,  0, -1), dtype=torch.float32)
    input_gfs  = torch.tensor(np.moveaxis(input_gfs,  0, -1), dtype=torch.float32)
    target     = torch.tensor(np.moveaxis(target,     0, -1), dtype=torch.float32)

    # Normalization statistics computed on training data only
    dataset.update(_compute_normalization_stats(input_sat, input_gfs, target, train_ids))

    # Cap maximum target value
    if dataset["ymax"] > MAX_PRECIP:
        dataset["ymax"] = MAX_PRECIP
        print(f"ymax capped at {MAX_PRECIP}")
    dataset["max_value"] = dataset["ymax"]

    dataset["TrainDataSet"] = _set_dataset(dataset, input_sat[:, :, train_ids],
                                           input_gfs[:, :, train_ids],
                                           target[:, :, train_ids])
    dataset["ValDataSet"]   = _set_dataset(dataset, input_sat[:, :, val_ids],
                                           input_gfs[:, :, val_ids],
                                           target[:, :, val_ids])
    dataset["TestDataSet"]  = _set_dataset(dataset, input_sat[:, :, test_ids],
                                           input_gfs[:, :, test_ids],
                                           target[:, :, test_ids])

    # Save split indices for reproducibility
    np.savez_compressed("output/train_ids.npz", train_ids=train_ids)
    np.savez_compressed("output/val_ids.npz",   val_ids=val_ids)
    np.savez_compressed("output/test_ids.npz",  test_ids=test_ids)

    return dataset


def _compute_normalization_stats(input_sat, input_gfs, target, train_ids):
    """Compute min, max, mean, and std for each input and target from training data."""
    return {
        "xmin":   input_sat[:, :, train_ids].min(),
        "xmax":   input_sat[:, :, train_ids].max(),
        "xmean":  torch.mean(input_sat[:, :, train_ids]),
        "xstd":   torch.std(input_sat[:, :, train_ids]),
        "xp_min": input_gfs[:, :, train_ids].min(),
        "xp_max": input_gfs[:, :, train_ids].max(),
        "xp_mean":torch.mean(input_gfs[:, :, train_ids]),
        "xp_std": torch.std(input_gfs[:, :, train_ids]),
        "ymin":   target[:, :, train_ids].min(),
        "ymax":   target[:, :, train_ids].max(),
        "ymean":  torch.mean(target[:, :, train_ids]),
        "ystd":   torch.std(target[:, :, train_ids]),
    }


def _build_splits(input_sat, input_gfs, target, n_file, nx, ny):
    """
    Apply quality filters and extract the fixed test set indices.

    Returns
    -------
    train_val_indices : np.ndarray
        Indices available for training and validation.
    test_ids : np.ndarray
        Fixed test set indices (Jan 2021 + Jun 2021).
    """
    nt = input_sat.shape[0]

    # Remove samples with NaN in input or target
    no_nan = np.sum(np.sum(np.isnan(target) + np.isnan(input_sat), 2), 1) == 0
    valid  = np.full(nt, False)
    valid[no_nan] = True
    valid[_INVALID_INDICES] = False

    # Extract fixed test set
    test_range  = np.append(
        np.arange(_TEST_RANGE_SUMMER[0], _TEST_RANGE_SUMMER[1] + 1),
        np.arange(_TEST_RANGE_WINTER[0], _TEST_RANGE_WINTER[1] + 1)
    )
    test_bool   = np.logical_and(valid, np.isin(n_file, test_range))
    test_ids    = np.where(test_bool)[0]

    # Remove a known problematic block from the test set
    test_ids = np.delete(test_ids, np.s_[2938:2958])

    valid[test_bool] = False  # Exclude test set from train/val pool

    # Filter by DPR rain coverage
    rain_mask = np.sum(np.sum(target > 0.1, 2), 1) > nx * ny * DPR_RAIN_THR

    # Filter by correlation between GOES-RRQPE and DPR
    corr = np.zeros(nt)
    for i in range(nt):
        corr[i] = np.corrcoef(target[i].reshape(-1), input_sat[i].reshape(-1))[0, 1]
    corr[np.isnan(corr)] = 0.0

    train_val_indices = np.where(
        np.logical_and(valid, np.logical_and(corr < CORRELATION_THR, rain_mask))
    )[0]

    print(f"Train+val samples after filtering: {len(train_val_indices)}")
    print(f"Test samples: {len(test_ids)}")

    return train_val_indices, test_ids


# ---------------------------------------------------------------------------
# PyTorch Dataset class
# ---------------------------------------------------------------------------

class _set_dataset(Dataset):
    """
    PyTorch Dataset wrapping GOES-16, GFS, and DPR arrays for use with DataLoader.

    Applies optional data transformations (log, power) and normalization
    ([0,1], [-1,1], or standardization) at construction time, using statistics
    computed from the training set only.
    """
    def __init__(self, data, input_sat, input_gfs, target):
        self.x_sat   = input_sat
        self.x_gfs   = input_gfs
        self.y       = target

        self.xmin,  self.xmax  = data["xmin"],  data["xmax"]
        self.xp_min, self.xp_max = data["xp_min"], data["xp_max"]
        self.ymin,  self.ymax  = data["ymin"],  data["ymax"]
        self.xmean, self.xstd  = data["xmean"], data["xstd"]
        self.xp_mean, self.xp_std = data["xp_mean"], data["xp_std"]
        self.ymean, self.ystd  = data["ymean"], data["ystd"]

        self.normalized = data["Norm"]
        self.typenorm   = data["TypeNorm"]
        self.device     = data["Device"]
        self.nx, self.ny = data["nx"], data["ny"]
        self.transform  = data["Transform"]

        # Optional data transformation applied before normalization
        if self.transform == "log":
            self.x_sat = torch.log(self.x_sat + 1.0)
            self.x_gfs = torch.log(self.x_gfs + 1.0)
            self.y     = torch.log(self.y     + 1.0)
        elif self.transform == "pow":
            self.x_sat = torch.pow(self.x_sat, 1.0 / 1.16)
            self.x_gfs = torch.pow(self.x_gfs, 1.0 / 1.16)
            self.y     = torch.pow(self.y,     1.0 / 1.16)

        # Normalization
        if self.normalized:
            if self.typenorm == "01":
                self.x_sat = (self.x_sat - self.xmin)  / (self.xmax  - self.xmin)
                self.x_gfs = (self.x_gfs - self.xp_min) / (self.xp_max - self.xp_min)
                self.y     = (self.y     - self.ymin)  / (self.ymax  - self.ymin)
            elif self.typenorm == "11":
                self.x_sat = 2 * (self.x_sat - self.xmin)  / (self.xmax  - self.xmin)  - 1
                self.x_gfs = 2 * (self.x_gfs - self.xp_min) / (self.xp_max - self.xp_min) - 1
                self.y     = 2 * (self.y     - self.ymin)  / (self.ymax  - self.ymin)  - 1
            elif self.typenorm == "standarized":
                self.x_sat = (self.x_sat - self.xmean) / self.xstd
                self.x_gfs = (self.x_gfs - self.xp_mean) / self.xp_std
                self.y     = (self.y     - self.ymean) / self.ystd

    def __len__(self):
        return self.x_sat.shape[2]

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise TypeError("Index must be an integer.")
        if index < 0 or index >= self.__len__():
            raise IndexError(f"Index {index} out of range (size {self.__len__()}).")
        return (self.x_sat[:, :, index].to(self.device),
                self.x_gfs[:, :, index].to(self.device),
                self.y[:, :, index].to(self.device))

    def get_data(self, numpy=False):
        """Return (input_sat, input_gfs, target) with time as the first axis."""
        sat = self.x_sat.moveaxis(2, 0)
        gfs = self.x_gfs.moveaxis(2, 0)
        tgt = self.y.moveaxis(2, 0)
        if numpy:
            return sat.numpy(), gfs.numpy(), tgt.numpy()
        return sat, gfs, tgt

    def get_data_device(self):
        """Return arrays moved to the configured device."""
        return (self.x_sat.moveaxis(2, 0).to(self.device),
                self.x_gfs.moveaxis(2, 0).to(self.device),
                self.y.moveaxis(2, 0).to(self.device))


# ---------------------------------------------------------------------------
# Denormalization utilities
# ---------------------------------------------------------------------------

def denorm(dset, input_sat, target, output):
    """Denormalize GOES input, target, and model output."""
    if not dset["Norm"]:
        return input_sat, target, output
    if dset["TypeNorm"] == "01":
        input_sat = input_sat * (dset["xmax"] - dset["xmin"]) + dset["xmin"]
        target    = target    * (dset["ymax"] - dset["ymin"]) + dset["ymin"]
        output    = output    * (dset["ymax"] - dset["ymin"]) + dset["ymin"]
    elif dset["TypeNorm"] == "11":
        input_sat = 0.5 * (input_sat + 1) * (dset["xmax"] - dset["xmin"]) + dset["xmin"]
        target    = 0.5 * (target    + 1) * (dset["ymax"] - dset["ymin"]) + dset["ymin"]
        output    = 0.5 * (output    + 1) * (dset["ymax"] - dset["ymin"]) + dset["ymin"]
    elif dset["TypeNorm"] == "standarized":
        input_sat = input_sat * dset["xstd"]  + dset["xmean"]
        target    = target    * dset["ystd"]  + dset["ymean"]
        output    = output    * dset["ystd"]  + dset["ymean"]
    return input_sat, target, output


def denormp(dset, input_gfs, target, output):
    """Denormalize GFS input, target, and model output."""
    if not dset["Norm"]:
        return input_gfs, target, output
    if dset["TypeNorm"] == "01":
        input_gfs = input_gfs * (dset["xp_max"] - dset["xp_min"]) + dset["xp_min"]
        target    = target    * (dset["ymax"]  - dset["ymin"])  + dset["ymin"]
        output    = output    * (dset["ymax"]  - dset["ymin"])  + dset["ymin"]
    elif dset["TypeNorm"] == "11":
        input_gfs = 0.5 * (input_gfs + 1) * (dset["xp_max"] - dset["xp_min"]) + dset["xp_min"]
        target    = 0.5 * (target    + 1) * (dset["ymax"]  - dset["ymin"])  + dset["ymin"]
        output    = 0.5 * (output    + 1) * (dset["ymax"]  - dset["ymin"])  + dset["ymin"]
    elif dset["TypeNorm"] == "standarized":
        input_gfs = input_gfs * dset["xp_std"]  + dset["xp_mean"]
        target    = target    * dset["ystd"]  + dset["ymean"]
        output    = output    * dset["ystd"]  + dset["ymean"]
    return input_gfs, target, output


def normalize_x(x, dset):
    """Normalize a GOES input array using training statistics."""
    if not dset["Norm"]:
        return x
    if dset["TypeNorm"] == "01":
        return (x - dset["xmin"]) / (dset["xmax"] - dset["xmin"])
    if dset["TypeNorm"] == "11":
        return 2 * (x - dset["xmin"]) / (dset["xmax"] - dset["xmin"]) - 1
    if dset["TypeNorm"] == "standarized":
        return (x - dset["xmean"]) / dset["xstd"]
    return x


def normalize_y(y, dset):
    """Normalize a target array using training statistics."""
    if not dset["Norm"]:
        return y
    if dset["TypeNorm"] == "01":
        return (y - dset["ymin"]) / (dset["ymax"] - dset["ymin"])
    if dset["TypeNorm"] == "11":
        return 2 * (y - dset["ymin"]) / (dset["ymax"] - dset["ymin"]) - 1
    if dset["TypeNorm"] == "standarized":
        return (y - dset["ymean"]) / dset["ystd"]
    return y


def denormalize_x(x, dset):
    """Denormalize a GOES input array."""
    if not dset["Norm"]:
        return x
    if dset["TypeNorm"] == "01":
        return x * (dset["xmax"] - dset["xmin"]) + dset["xmin"]
    if dset["TypeNorm"] == "11":
        return 0.5 * (x + 1) * (dset["xmax"] - dset["xmin"]) + dset["xmin"]
    if dset["TypeNorm"] == "standarized":
        return x * dset["xstd"] + dset["xmean"]
    return x


def denormalize_y(y, dset):
    """Denormalize a target or output array."""
    if not dset["Norm"]:
        return y
    if dset["TypeNorm"] == "01":
        return y * (dset["ymax"] - dset["ymin"]) + dset["ymin"]
    if dset["TypeNorm"] == "11":
        return 0.5 * (y + 1) * (dset["ymax"] - dset["ymin"]) + dset["ymin"]
    if dset["TypeNorm"] == "standarized":
        return y * dset["ystd"] + dset["ymean"]
    return y


# ---------------------------------------------------------------------------
# Test set evaluation
# ---------------------------------------------------------------------------

def evaluate_on_test(model, dataset, plot_path):
    """
    Evaluate a trained model on the test set and print RMSE, bias, and
    Pearson correlation. Saves a scatter plot of predictions vs. targets.

    Parameters
    ----------
    model : torch.nn.Module
    dataset : dict
        Dataset dictionary (output of get_Data).
    plot_path : str
        Directory where the scatter plot will be saved.
    """
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset["TestDataSet"], batch_size=dataset["BatchSize"], shuffle=False
    )

    preds, targets = [], []
    with torch.no_grad():
        for x_sat, x_gfs, y in loader:
            preds.append(model(x_sat, x_gfs).cpu())
            targets.append(y.cpu())

    preds   = torch.cat(preds,   dim=0)
    targets = torch.cat(targets, dim=0)

    print(f"Normalized output range: "
          f"[{preds.numpy().min():.4f}, {preds.numpy().max():.4f}]")

    _, targets, preds = denorm(dataset, preds, targets, preds)
    y_true = targets.numpy().reshape(-1)
    y_pred = preds.numpy().reshape(-1)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    bias = np.mean(y_pred - y_true)
    corr = pearsonr(y_true, y_pred)[0]
    print(f"Test RMSE: {rmse:.4f} | BIAS: {bias:.4f} | CORR: {corr:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.4, color="royalblue", s=5, label="Test")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
             "r--", label="Ideal")
    plt.xlabel("Target [mm/h]")
    plt.ylabel("Model output [mm/h]")
    plt.title("Test set: target vs. prediction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{plot_path}/test_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Scatter plot saved to {plot_path}/test_scatter.png")


def evaluate_bootstrap(model, dataset, n_bootstrap, plot_path):
    """
    Estimate uncertainty in test set metrics via bootstrap resampling over images.

    For each bootstrap replicate, images are sampled with replacement and RMSE,
    bias, and Pearson correlation are computed. Reports the 5th and 95th percentiles
    and saves histograms of each metric.

    Parameters
    ----------
    model : torch.nn.Module
    dataset : dict
    n_bootstrap : int
        Number of bootstrap replicates.
    plot_path : str
        Directory where bootstrap histograms will be saved.

    Returns
    -------
    dict with keys 'rmse_list', 'bias_list', 'corr_list'
    """
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset["TestDataSet"], batch_size=dataset["BatchSize"], shuffle=False
    )

    preds, targets = [], []
    with torch.no_grad():
        for x_sat, x_gfs, y in loader:
            preds.append(model(x_sat, x_gfs).cpu())
            targets.append(y.cpu())

    preds   = torch.cat(preds,   dim=0)
    targets = torch.cat(targets, dim=0)
    _, targets, preds = denorm(dataset, preds, targets, preds)

    y_true = targets.numpy()   # shape (N, H, W)
    y_pred = preds.numpy()
    n_imgs = y_true.shape[0]

    rmse_list, bias_list, corr_list = [], [], []
    for _ in range(n_bootstrap):
        idx   = np.random.choice(n_imgs, size=n_imgs, replace=True)
        yt    = y_true[idx].reshape(-1)
        yp    = y_pred[idx].reshape(-1)
        rmse_list.append(np.sqrt(np.mean((yt - yp) ** 2)))
        bias_list.append(np.mean(yp - yt))
        corr_list.append(pearsonr(yt, yp)[0])

    for metric, values, color, label in [
        ("rmse", rmse_list, "skyblue",   "RMSE"),
        ("bias", bias_list, "orange",    "Bias"),
        ("corr", corr_list, "limegreen", "Pearson correlation"),
    ]:
        p5, p95 = np.percentile(values, [5, 95])
        print(f"{label}: P5={p5:.4f}  P95={p95:.4f}")
        plt.figure(figsize=(8, 5))
        plt.hist(values, bins=20, color=color, edgecolor="black", alpha=0.7)
        plt.title(f"Bootstrap distribution — {label}")
        plt.xlabel(label)
        plt.ylabel("Frequency")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{plot_path}/bootstrap_{metric}.png", dpi=300)
        plt.close()
        print(f"  Saved: {plot_path}/bootstrap_{metric}.png")

    return {"rmse_list": rmse_list, "bias_list": bias_list, "corr_list": corr_list}