"""
Visualization functions for the DPR-GPM precipitation retrieval model.

Produces the following plots (all saved to conf['PlotPath']):
    - Training summary : loss curves (RMSE, bias, correlation, total loss)
                         and categorical skill scores (POD, FAR, ETS) on
                         validation and test sets across precipitation thresholds.
    - Distribution histograms : log-frequency distributions of target vs.
                                 model output for train, validation, and test sets.
    - Spatial mean fields : mean and occurrence frequency maps of input,
                            target, and output fields for all splits.
    - Scatter plot : hexbin density plot of target vs. model output for all splits.
    - Individual cases : 4-panel figures (GOES input, GFS input, DPR target,
                         model output) for selected high-precipitation events.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm

import models
import dataset as ds
import verification as ver


# ---------------------------------------------------------------------------
# Precipitation thresholds used for categorical skill scores [mm/h]
# ---------------------------------------------------------------------------
THRESHOLDS = [0.1, 0.5, 1, 3, 5, 10, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150]

# Minimum criteria for selecting high-precipitation cases to plot
CASE_MAX_THR  = 40.0   # minimum peak rain rate [mm/h]
CASE_MEAN_THR = 3.0    # minimum mean rain rate [mm/h]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def plotting(dataset, model, conf, scores):
    """
    Run all diagnostic plots for training, validation, and test sets.

    Evaluates the model on all three splits, denormalizes all fields,
    and calls each plotting function in sequence.

    Parameters
    ----------
    dataset : dict
        Output of ds.get_Data().
    model : torch.nn.Module
        Trained model.
    conf : dict
        Training configuration (must contain 'PlotPath', 'Input_Sat_Name',
        'Input_Prono_Name', 'TargetVarName', 'BatchSize', 'PlotCasesList').
    scores : dict
        Training history (output of models.model_train).
    """
    os.makedirs(conf["PlotPath"], exist_ok=True)
    data = {}

    for split in ["Train", "Val", "Test"]:
        sat, gfs, targets, outputs = models.model_eval(
            model, dataset[split + "DataSet"], conf["BatchSize"], numpy=False
        )
        sat, targets, outputs = ds.denorm(dataset, sat, targets, outputs)
        gfs, _, _             = ds.denormp(dataset, gfs, targets, outputs)

        sat     = sat.cpu().detach().numpy()
        gfs     = gfs.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        outputs = outputs.cpu().detach().numpy()

        data[f"vmin_input"],  data[f"vmax_input"]  = np.nanmin(sat),     np.nanmax(sat)
        data[f"vmin_target"], data[f"vmax_target"]  = np.nanmin(targets), np.nanmax(targets)
        data[f"vmin_gfs"],    data[f"vmax_gfs"]     = np.nanmin(gfs),     np.nanmax(gfs)

        data[f"{split}_input"]  = sat
        data[f"{split}_gfs"]    = gfs
        data[f"{split}_target"] = targets
        data[f"{split}_output"] = outputs

        print(f"[{split}] output range after denormalization: "
              f"[{np.nanmin(outputs):.4f}, {np.nanmax(outputs):.4f}]")

    plot_training_summary(data, scores, conf)
    plot_distribution_histograms(data, conf)
    plot_mean_fields(data, conf)
    plot_scatter(data, conf)
    plot_cases(data, conf)


# ---------------------------------------------------------------------------
# Training summary
# ---------------------------------------------------------------------------

def plot_training_summary(data, scores, conf):
    """
    Save a figure with loss curves (RMSE, bias, correlation, total loss) for
    train and validation sets, plus POD/FAR/ETS skill scores on both the
    validation and test sets across precipitation thresholds.

    The loss curves are truncated at the best validation RMSE epoch.

    Output: <PlotPath>/training_summary.png
    """
    best_epoch = np.argmin(scores["ValRmse"])

    fig = plt.figure(figsize=(20, 12))

    # --- Loss curves (train vs val) ---
    for subplot, key, title in [
        (231, "Rmse", "RMSE"),
        (232, "Bias", "Bias"),
        (234, "Corr", "Pearson Correlation"),
        (235, "Loss", "Loss"),
    ]:
        ax = fig.add_subplot(subplot)
        ax.plot(scores[f"Train{key}"][:best_epoch], "-r", label="Train")
        ax.plot(scores[f"Val{key}"][:best_epoch],   "-b", label="Val")
        if key == "Rmse":
            ax.axvline(best_epoch, color="orange", linestyle="--",
                       label=f"Best epoch {best_epoch} "
                             f"(RMSE={scores['ValRmse'][best_epoch]:.2f})")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid()

    # --- Categorical skill scores: val and test side by side ---
    ax = fig.add_subplot(133)
    x  = range(len(THRESHOLDS))

    for split, linestyle in [("Val", "--"), ("Test", "-")]:
        cf = ver.confusion_matrix_thresholds(
            data[f"{split}_output"], data[f"{split}_target"], THRESHOLDS
        )
        for metric_fn, color, label in [
            (ver.pod_score, "green", "POD"),
            (ver.far_score, "red",   "FAR"),
            (ver.ets_score, "blue",  "ETS"),
        ]:
            values = metric_fn(cf)
            ax.plot(x, values, color=color, linestyle=linestyle,
                    label=f"{label} ({split})")
            ax.scatter(x, values, color=color, s=15)

    ax.set_title("Categorical scores — Val (--) vs. Test (—)", fontsize=13)
    ax.set_xlabel("Rain rate [mm/h]")
    ax.set_xticks(x)
    ax.set_xticklabels(THRESHOLDS, rotation=45, fontsize=8)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=2)
    ax.grid()

    path = os.path.join(conf["PlotPath"], "training_summary.png")
    fig.savefig(path, dpi="figure", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Distribution histograms
# ---------------------------------------------------------------------------

def plot_distribution_histograms(data, conf):
    """
    Save log-frequency distribution plots comparing the target and model output
    precipitation distributions for train, validation, and test sets.

    Output: <PlotPath>/distribution_histograms.png
    """
    target_name = conf["TargetVarName"]
    vmin, vmax, _, _, label, unit, _, _, _, _, _, bins, dticks = \
        _plot_config_target(target_name)

    splits = ["Train", "Val", "Test"]
    fig, axes = plt.subplots(1, len(splits), figsize=(24, 6))
    fig.suptitle(f"Log-frequency distribution: model output vs. {target_name}")

    for ax, split in zip(axes, splits):
        hist_out, _ = np.histogram(data[f"{split}_output"].flatten(), bins)
        hist_tgt, _ = np.histogram(data[f"{split}_target"].flatten(), bins)
        hist_out = hist_out / hist_out.sum()
        hist_tgt = hist_tgt / hist_tgt.sum()

        ax.plot(bins[:-1], np.log(hist_tgt), "-b", label=f"{target_name} (target)")
        ax.plot(bins[:-1], np.log(hist_out), "-r", label="Model output")
        ax.set_xticks(np.arange(vmin, vmax + dticks, dticks))
        ax.set_xlabel(f"{label} {unit}")
        ax.set_ylabel("Log(frequency)")
        ax.set_title(split)
        ax.legend()
        ax.grid()

    path = os.path.join(conf["PlotPath"], "distribution_histograms.png")
    fig.savefig(path, dpi="figure", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Mean fields
# ---------------------------------------------------------------------------

def plot_mean_fields(data, conf):
    """
    Save spatial mean and occurrence frequency maps for input, target,
    and model output fields, one figure per data split.

    Outputs: <PlotPath>/mean_<split>.png, <PlotPath>/count_<split>.png
    """
    input_name  = conf["Input_Sat_Name"]
    target_name = conf["TargetVarName"]

    _, _, _, _, input_label, unit_input, cmap_input, _, bounds_mean_input, \
        _, norm_mean_input, _ = _plot_config_input(input_name)
    _, _, _, _, target_label, unit_target, cmap_target, _, bounds_mean_target, \
        _, norm_mean_target, _, _ = _plot_config_target(target_name)

    for split in ["Train", "Val", "Test"]:
        _, nx, ny = data[f"{split}_input"].shape

        # Mean fields
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(f"Mean fields — {split}")

        for ax, field, label, unit, cmap, bounds, norm in [
            (axes[0], data[f"{split}_input"],  input_label,  unit_input,
             cmap_input,  bounds_mean_input,  norm_mean_input),
            (axes[1], data[f"{split}_target"], target_label, unit_target,
             cmap_target, bounds_mean_target, norm_mean_target),
            (axes[2], data[f"{split}_output"], target_label, unit_target,
             cmap_target, bounds_mean_target, norm_mean_target),
        ]:
            cm = ax.pcolor(np.mean(field, axis=0), norm=norm, cmap=cmap)
            cbar = fig.colorbar(cm, ax=ax)
            cbar.set_ticks(bounds)
            cbar.set_label(f"{label} {unit}")
            ax.set_xticks(np.arange(0, nx + 4, 4))
            ax.set_yticks(np.arange(0, ny + 4, 4))
            ax.grid()

        axes[0].set_title(f"GOES input: {input_name}")
        axes[1].set_title(f"DPR target: {target_name}")
        axes[2].set_title("Model output")

        path = os.path.join(conf["PlotPath"], f"mean_{split}.png")
        fig.savefig(path, dpi="figure", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

        # Occurrence frequency
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(f"Occurrence frequency (> 0) — {split}")

        counts = [
            np.sum(data[f"{split}_input"]  > 0, axis=0),
            np.sum(data[f"{split}_target"] > 0, axis=0),
            np.sum(data[f"{split}_output"] > 0, axis=0),
        ]
        vmin_c = min(c.min() for c in counts)
        vmax_c = max(c.max() for c in counts)

        for ax, count, title in zip(axes, counts, ["GOES input", "DPR target", "Model output"]):
            cm = ax.pcolor(count, cmap="nipy_spectral_r", vmin=vmin_c, vmax=vmax_c)
            fig.colorbar(cm, ax=ax)
            ax.set_title(title)

        path = os.path.join(conf["PlotPath"], f"count_{split}.png")
        fig.savefig(path, dpi="figure", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Scatter plot
# ---------------------------------------------------------------------------

def plot_scatter(data, conf):
    """
    Save hexbin scatter plots of DPR target vs. model output for train,
    validation, and test sets, annotated with RMSE, bias, and Pearson
    correlation.

    Output: <PlotPath>/scatter_<split>.png
    """
    target_name = conf["TargetVarName"]
    vmin, vmax, _, _, label, unit, _, _, _, _, _, _, dticks = \
        _plot_config_target(target_name)

    for split in ["Train", "Val", "Test"]:
        y_true = data[f"{split}_target"].flatten()
        y_pred = data[f"{split}_output"].flatten()

        rmse = ver.rmse(y_pred,        y_true)
        bias = ver.bias(y_pred,        y_true)
        corr = ver.pearson_corr(y_pred, y_true)

        fig, ax = plt.subplots(figsize=(8, 8))
        hb = ax.hexbin(y_true, y_pred, cmap="gist_ncar_r", bins="log",
                       gridsize=50, extent=(vmin, vmax, vmin, vmax))
        ax.plot([vmin, vmax], [vmin, vmax], "w--", linewidth=1)
        ax.set_xlabel(f"{label} {unit}")
        ax.set_ylabel(f"Model {label} {unit}")
        ax.set_title(f"Target vs. model output — {split}\n"
                     f"RMSE={rmse:.2f}  BIAS={bias:.2f}  Corr={corr:.3f}")
        ax.set_xticks(np.arange(vmin, vmax + dticks, dticks))
        ax.set_yticks(np.arange(vmin, vmax + dticks, dticks))
        ax.grid()

        cbar = fig.colorbar(hb, ax=ax, orientation="vertical")
        cbar.set_label("Frequency")
        cbar.set_ticks([1, 10, 100, 1000, 10000, 100000])
        cbar.set_ticklabels([r"$10^0$", r"$10^1$", r"$10^2$",
                              r"$10^3$", r"$10^4$", r"$10^5$"])

        path = os.path.join(conf["PlotPath"], f"scatter_{split}.png")
        fig.savefig(path, dpi="figure", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Individual case panels
# ---------------------------------------------------------------------------

def plot_cases(data, conf):
    """
    Save 4-panel figures (GOES input, GFS input, DPR target, model output)
    for a set of selected images from train, validation, and test sets.

    For train and val, images are selected by:
        1. Uniformly sampled fractions of the dataset (conf['PlotCasesList']).
        2. All images where peak rain rate > CASE_MAX_THR and
           mean rain rate > CASE_MEAN_THR.

    For the test set, only criterion 2 is applied (all high-precipitation
    events are plotted, without uniform sampling).

    Output: <PlotPath>/cases/<split>_case_<index>.png
    """
    input_name  = conf["Input_Sat_Name"]
    gfs_name    = conf["Input_Prono_Name"]
    target_name = conf["TargetVarName"]

    _, _, _, _, input_label,  unit_input,  cmap_input,  bounds_input, \
        _, norm_input,  _, _ = _plot_config_input(input_name)
    _, _, _, _, gfs_label,    unit_gfs,    cmap_gfs,    bounds_gfs, \
        _, norm_gfs,    _, _ = _plot_config_input(gfs_name)
    _, _, _, _, target_label, unit_target, cmap_target, bounds_target, \
        _, norm_target, _, _, _ = _plot_config_target(target_name)

    cases_dir = os.path.join(conf["PlotPath"], "cases")
    os.makedirs(cases_dir, exist_ok=True)

    for split in ["Train", "Val", "Test"]:
        nt, nx, ny = data[f"{split}_input"].shape

        # For test, plot all high-precipitation events without uniform sampling
        if split == "Test":
            plot_indices = set()
        else:
            plot_indices = set((conf["PlotCasesList"] * nt).astype(int).tolist())

        for img in range(nt):
            tgt = data[f"{split}_target"][img]
            if np.max(tgt) > CASE_MAX_THR and np.mean(tgt) > CASE_MEAN_THR:
                plot_indices.add(img)

        print(f"[{split}] Plotting {len(plot_indices)} cases.")

        for idx in sorted(plot_indices):
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            fig.suptitle(f"{split} — image {idx}", fontsize=13)

            panels = [
                (data[f"{split}_input"][idx],  norm_input,  cmap_input,
                 bounds_input,  f"GOES input: {input_label} {unit_input}"),
                (data[f"{split}_gfs"][idx],    norm_gfs,    cmap_gfs,
                 bounds_gfs,    f"GFS input: {gfs_label} {unit_gfs}"),
                (data[f"{split}_target"][idx], norm_target, cmap_target,
                 bounds_target, f"DPR target: {target_label} {unit_target}"),
                (data[f"{split}_output"][idx], norm_target, cmap_target,
                 bounds_target, f"Model output: {target_label} {unit_target}"),
            ]

            for ax, (field, norm, cmap, bounds, title) in zip(axes, panels):
                cm = ax.pcolor(field, norm=norm, cmap=cmap)
                cbar = fig.colorbar(cm, ax=ax, spacing="uniform")
                cbar.set_ticks(bounds)
                ax.set_xticks(np.arange(0, nx + 4, 4))
                ax.set_yticks(np.arange(0, ny + 4, 4))
                ax.set_title(title, fontsize=10)
                ax.grid()

            path = os.path.join(cases_dir, f"{split}_case_{idx}.png")
            fig.savefig(path, dpi="figure", bbox_inches="tight")
            plt.close(fig)

        print(f"[{split}] Cases saved to {cases_dir}")


# ---------------------------------------------------------------------------
# Plot configuration helpers
# ---------------------------------------------------------------------------

def _plot_config_input(variable_name):
    """
    Return display configuration for a given input variable.

    Returns
    -------
    vmin, vmax, vmin_mean, vmax_mean, label, unit,
    cmap, bounds, bounds_mean, norm, norm_mean, dticks
    """
    configs = {
        "TB": {
            "vmin": 183, "vmax": 313, "vmin_mean": 250, "vmax_mean": 275,
            "label": "Cloud Top Temperature", "unit": "[K]", "dticks": 10,
        },
        "RRQPE": {
            "vmin": 0, "vmax": 100, "vmin_mean": 0, "vmax_mean": 3,
            "label": "RRQPE", "unit": "[mm/h]", "dticks": 25,
        },
        "pwat": {
            "vmin": 0, "vmax": 70, "vmin_mean": 0, "vmax_mean": 70,
            "label": "Precipitable water", "unit": "[kg m⁻²]", "dticks": 10,
        },
        "cape": {
            "vmin": 0, "vmax": 5000, "vmin_mean": 0, "vmax_mean": 5000,
            "label": "CAPE", "unit": "[J kg⁻¹]", "dticks": 500,
        },
    }

    if variable_name not in configs:
        vmin, vmax = 0, 1
        norm   = plt.Normalize(vmin, vmax)
        cmap   = get_cmap("viridis")
        bounds = np.linspace(vmin, vmax, 11)
        return (vmin, vmax, vmin, vmax, variable_name, "",
                cmap, bounds, bounds, norm, norm, 0.1)

    c = configs[variable_name]
    cmap, bounds, bounds_mean, norm, norm_mean = _build_colormap(
        variable_name, c["vmin"], c["vmax"], c["vmin_mean"], c["vmax_mean"]
    )
    return (c["vmin"], c["vmax"], c["vmin_mean"], c["vmax_mean"],
            c["label"], c["unit"], cmap, bounds, bounds_mean,
            norm, norm_mean, c["dticks"])


def _plot_config_target(variable_name):
    """
    Return display configuration for the target variable.

    Returns
    -------
    vmin, vmax, vmin_mean, vmax_mean, label, unit,
    cmap, bounds, bounds_mean, norm, norm_mean, bins, dticks
    """
    if variable_name == "PP_DPR":
        vmin, vmax           = 0, 275
        vmin_mean, vmax_mean = 0, 3
        label, unit          = "Rain Rate", "[mm/h]"
        dticks               = 25
        bins                 = np.arange(vmin, vmax + 1, 1)
        cmap, bounds, bounds_mean, norm, norm_mean = _build_colormap(
            variable_name, vmin, vmax, vmin_mean, vmax_mean
        )
        return (vmin, vmax, vmin_mean, vmax_mean, label, unit,
                cmap, bounds, bounds_mean, norm, norm_mean, bins, dticks)

    raise ValueError(f"Unknown target variable: {variable_name}")


def _build_colormap(variable, vmin, vmax, vmin_mean, vmax_mean):
    """
    Build a custom colormap, boundary array, and BoundaryNorm for a variable.

    Returns
    -------
    cmap, bounds, bounds_mean, norm, norm_mean
    """
    gist_ncar_r = get_cmap("gist_ncar_r", 256)
    greys       = get_cmap("Greys",       256)
    gist_ncar   = get_cmap("gist_ncar",   256)

    if variable in ("RRQPE", "PP_DPR"):
        bounds     = [0, 0.1, 0.5, 1, 5, 10, 25, 40, 60, 80, 100, 150, 200, 250]
        rgb_colors = [
            [0, 0, 0],     [0, 0, 255],   [0, 203, 255],
            [0, 255, 0],   [255, 255, 0], [255, 127, 0],
            [255, 0, 0],   [127, 0, 0],   [255, 0, 255],
            [64, 0, 127],  [255, 255, 255],
        ]
        colors      = [[v / 255 for v in c] for c in rgb_colors]
        cmap        = LinearSegmentedColormap.from_list("rainrate", colors, N=256)
        norm        = BoundaryNorm(bounds, ncolors=256)
        bounds_mean = np.arange(vmin_mean, vmax_mean + 0.2, 0.2)
        norm_mean   = BoundaryNorm(bounds_mean, ncolors=256)
        return cmap, bounds, bounds_mean, norm, norm_mean

    if variable == "TB":
        combined = np.concatenate((
            gist_ncar_r(np.linspace(0.1, 1, gist_ncar_r.N)),
            greys(np.linspace(0, 1, greys.N))
        ), axis=0)
        cmap        = ListedColormap(combined.tolist())
        bounds      = np.arange(vmin, vmax + 5, 5)
        bounds_mean = np.arange(vmin_mean, vmax_mean + 3, 3)
        norm        = BoundaryNorm(bounds,      ncolors=512)
        norm_mean   = BoundaryNorm(bounds_mean, ncolors=512)
        return cmap, bounds, bounds_mean, norm, norm_mean

    if variable in ("pwat", "cape"):
        threshold  = 20
        grey_part  = greys(np.linspace(0.4, 1, 64))
        ncar_part  = gist_ncar(np.linspace(0, 1, 192))
        combined   = np.vstack([grey_part, ncar_part])
        cmap       = ListedColormap(combined)
        bounds     = np.concatenate((
            np.linspace(0, threshold, len(grey_part) + 1),
            np.linspace(threshold + 1e-5, vmax, len(ncar_part))
        ))
        bounds_mean = np.arange(vmin_mean, vmax_mean + 3, 3)
        norm        = BoundaryNorm(bounds,      ncolors=len(combined))
        norm_mean   = BoundaryNorm(bounds_mean, ncolors=len(bounds_mean))
        return cmap, bounds, bounds_mean, norm, norm_mean

    # Fallback: linear colormap
    bounds      = np.linspace(vmin, vmax, 11)
    bounds_mean = np.linspace(vmin_mean, vmax_mean, 11)
    norm        = BoundaryNorm(bounds,      ncolors=256)
    norm_mean   = BoundaryNorm(bounds_mean, ncolors=256)
    return get_cmap("viridis"), bounds, bounds_mean, norm, norm_mean