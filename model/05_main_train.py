"""
Training script for the DPR-GPM precipitation retrieval model.

This script loads the input data (GOES-16 brightness temperatures and GFS
forecast variables), configures the model and training hyperparameters from
a experiment configuration function, trains the model, evaluates it on the
test set, and saves all outputs (model weights, plots, dataset, scores).

Usage
-----
    python 05_main_train.py <loss_name> <exp_number> <gfs_variable>

Arguments
---------
    loss_name    : Name of the loss function experiment (e.g. MSE, MSE_SF, Quantile,
                   MSE_inv_PDF, MultiLoss). Must match a function defined in
                   conf_experimentos.py as exp_<loss_name>_<exp_number>.
    exp_number   : Experiment number within the loss family (e.g. 1, 2, 3).
    gfs_variable : Name of the GFS variable to use as forecast input
                   (e.g. pwat, cape, gh850). Must match a key in the
                   interpolated dataset.

Example
-------
    python 05_main_train.py MSE 1 pwat
"""

import os
import sys
import gc
import pickle

import numpy as np
import torch

import dataset as ds
import models as models
import plots
import conf
import loss_functions as LF
import verification

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

models.define_seed(seed=1024)

# ---------------------------------------------------------------------------
# Parse command-line arguments
# ---------------------------------------------------------------------------

args = sys.argv
if len(args) < 4:
    raise SystemExit(
        "Usage: python 05_main_train.py <loss_name> <exp_number> <gfs_variable>"
    )

loss_name    = args[1]
exp_number   = args[2]
gfs_variable = args[3]

# ---------------------------------------------------------------------------
# Load experiment configuration
# ---------------------------------------------------------------------------

config_call = f"conf_experimentos.exp_{loss_name}_{exp_number}()"
print(f"Loading configuration: {config_call}")

# Unpack hyperparameters — the number of return values depends on the loss family
if loss_name == "MSE_SF":
    (ExpName, ExpNumber, ModelType, ActType, OutActType, init_dist, gain_function,
     BatchSize, MaxEpochs, Norm, TypeNorm, Transform, LearningRate, scheduler_act,
     Milestones, Gamma, WeightDecay, dropout_act, alpha) = eval(config_call)

elif loss_name == "MSE_inv_PDF":
    (ExpName, ExpNumber, ModelType, ActType, OutActType, init_dist, gain_function,
     BatchSize, MaxEpochs, Norm, TypeNorm, Transform, LearningRate, scheduler_act,
     Milestones, Gamma, WeightDecay, dropout_act, max_weight, weight_factor) = eval(config_call)

elif loss_name in ("Quantile", "MultiLoss"):
    (ExpName, ExpNumber, ModelType, ActType, OutActType, init_dist, gain_function,
     BatchSize, MaxEpochs, Norm, TypeNorm, Transform, LearningRate, scheduler_act,
     Milestones, Gamma, WeightDecay, dropout_act, alpha) = eval(config_call)

else:
    (ExpName, ExpNumber, ModelType, ActType, OutActType, init_dist, gain_function,
     BatchSize, MaxEpochs, Norm, TypeNorm, Transform, LearningRate, scheduler_act,
     Milestones, Gamma, WeightDecay, dropout_act) = eval(config_call)

# ---------------------------------------------------------------------------
# Build configuration dictionary
# ---------------------------------------------------------------------------

conf = {
    # Experiment identifiers
    "ExpName":           ExpName,
    "ExpNumber":         ExpNumber,

    # Hardware
    "Device":            "cuda" if torch.cuda.is_available() else "cpu",

    # Input data
    "Input_Prono_Name":  gfs_variable,
    "DataFileInput_Prono":  f"path/to/interpolated_data/{gfs_variable}_interpolated.npz",    
    "DataFileInput_Sat": "path/to/goes_data/",
    "Input_Sat_Name":    "TB",

    # Target data
    "DataFileTarget":    "path/to/dpr_data/",
    "TargetVarName":     "PP_DPR",

    # N-file index (maps dataset rows to original GPM image indices)
    "NfileVarName":      "N_file",
    "DataFileN":         "path/to/N_sNAN.npz",

    # Model architecture
    "ModelType":         ModelType,
    "ActType":           ActType,
    "OutActType":        OutActType,
    "dropout_act":       dropout_act,
    "init_dist":         init_dist,
    "gain_function":     gain_function,

    # Training hyperparameters
    "BatchSize":         BatchSize,
    "MaxEpochs":         MaxEpochs,
    "Norm":              Norm,
    "TypeNorm":          TypeNorm,
    "Transform":         Transform,
    "Optimizer":         torch.optim.Adam,
    "Scheduler":         torch.optim.lr_scheduler.MultiStepLR,
    "LearningRate":      LearningRate,
    "scheduler_act":     scheduler_act,
    "Milestones":        Milestones,
    "Gamma":             Gamma,
    "WeightDecay":       WeightDecay,

    # Train / validation / test split
    "TrainRatio":        0.8,
    "ValRatio":          0.1,
    "Shuffle":           True,

    # Plotting — fractions of the dataset used to select cases to plot
    "PlotCases":         True,
    "PlotCasesList":     np.arange(0, 1, 0.05),

    # Output
    "SaveDataSet":       True,
}

# Instantiate model (architecture defined in models.py)
conf["Model"] = getattr(models, conf["ModelType"])(conf)

# ---------------------------------------------------------------------------
# Resolve output directory
# Iterates up to MAX_RUNS times to avoid overwriting previous experiments
# with the same configuration (e.g. different random seeds).
# ---------------------------------------------------------------------------

BASE_OUTPUT = "output/models"
MAX_RUNS    = 10

os.makedirs(BASE_OUTPUT, exist_ok=True)

for run_index in range(MAX_RUNS):
    candidate = os.path.join(BASE_OUTPUT, str(run_index))
    if not os.path.exists(candidate):
        conf["OutPath"] = candidate + "/"
        print(f"Output directory: {conf['OutPath']}")
        break
    print(f"Directory {candidate} already exists, trying next.")
else:
    raise SystemExit(
        f"All {MAX_RUNS} output slots for this configuration are occupied. "
        "Clean up old runs or increase MAX_RUNS."
    )

conf["PlotPath"] = os.path.join(conf["OutPath"], "figures/")
os.makedirs(conf["OutPath"],  exist_ok=True)
os.makedirs(conf["PlotPath"], exist_ok=True)
print("Output directories created.")

# ---------------------------------------------------------------------------
# Instantiate loss function
# ---------------------------------------------------------------------------

if loss_name == "MSE_SF":
    conf["alpha"] = alpha
    conf["Loss"]  = LF.MSE_Softmax(conf["alpha"])

elif loss_name == "MSE_inv_PDF":
    # Weights are derived from the training data distribution
    a, b = LF.get_weights(None, max_weight)   # DataSet not yet built; update if needed
    print(f"MSE_inv_PDF weights — a: {a:.4f}, b: {b:.4f}, "
          f"max_weight: {max_weight}, weight_factor: {weight_factor}")
    conf["Loss"] = LF.MSE_inv_PDF(a, b, max_weight, None, weight_factor)

elif loss_name == "Quantile":
    conf["alpha"] = alpha
    conf["Loss"]  = LF.Quantile_loss(alpha=conf["alpha"])

elif loss_name == "MultiLoss":
    conf["Loss"] = LF.MultiLoss()

else:
    conf["Loss"] = LF.MSE()

# ---------------------------------------------------------------------------
# Load input data
# ---------------------------------------------------------------------------

# GOES-16 brightness temperatures
InputData_Sat = np.load(conf["DataFileInput_Sat"])[conf["Input_Sat_Name"]]

# GFS forecast variable
InputData_GFS = np.load(conf["DataFileInput_Prono"])[gfs_variable]
print(f"GFS input variable: {gfs_variable}, shape: {InputData_GFS.shape}")

# DPR precipitation target
TargetData = np.load(conf["DataFileTarget"])[conf["TargetVarName"]]

# N-file index: maps each sample to its original GPM image
NfileData = np.load(conf["DataFileN"])[conf["NfileVarName"]]

# ---------------------------------------------------------------------------
# Build dataset and dataloaders
# ---------------------------------------------------------------------------

DataSet = ds.get_Data(InputData_Sat, InputData_GFS, TargetData, NfileData, conf)

# MSE_inv_PDF weights require the dataset — recompute now that it is available
if loss_name == "MSE_inv_PDF":
    a, b = LF.get_weights(DataSet, max_weight)
    print(f"MSE_inv_PDF weights (recomputed) — a: {a:.4f}, b: {b:.4f}")
    conf["Loss"] = LF.MSE_inv_PDF(a, b, max_weight, DataSet, weight_factor)

gc.collect()

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

model, scores = models.model_train(DataSet, conf)

# ---------------------------------------------------------------------------
# Evaluate best model on test set
# ---------------------------------------------------------------------------

best_model_path = os.path.join(
    conf["OutPath"], f"best_model_exp_{conf['ExpNumber']}.pth"
)
model.load_state_dict(torch.load(best_model_path))
print("=" * 80)
print("EVALUATING BEST MODEL ON TEST SET")
print("=" * 80)

ds.evaluate_on_test(model, DataSet, conf["PlotPath"])
plots.plotting(DataSet, model, conf, scores)

gc.collect()

# ---------------------------------------------------------------------------
# Save scripts, dataset, scores, and configuration
# ---------------------------------------------------------------------------

# Copy scripts used in this run for reproducibility
for script in [os.path.basename(__file__), "dataset.py"]:
    os.system(f"cp {script} {conf['OutPath']}")

if conf["SaveDataSet"]:
    for obj, name in [(DataSet, "DataSet"), (scores, "Scores"), (conf, "Conf")]:
        path = os.path.join(conf["OutPath"], f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved: {path}")

gc.collect()
print("Training complete.")