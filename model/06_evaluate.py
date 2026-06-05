"""
Post-training evaluation script.

Loads saved model weights, dataset, configuration, and scores from a
training output directory and regenerates all diagnostic plots.

Usage
-----
    python 06_evaluate.py <exp_dir>

Arguments
---------
    exp_dir : path to the output directory of a training run
              (e.g. output/models/0). Must contain:
              - DataSet.pkl
              - Conf.pkl
              - Scores.pkl
              - best_model_exp_<N>.pth

Example
-------
    python 06_evaluate.py output/models/3
"""

import sys
import os
import pickle
import torch
import plots

def load_experiment(exp_dir):
    """
    Load dataset, configuration, scores, and best model from an output directory.

    Parameters
    ----------
    exp_dir : str

    Returns
    -------
    dataset, conf, scores, model : loaded objects
    """
    for filename in ("DataSet.pkl", "Conf.pkl", "Scores.pkl"):
        path = os.path.join(exp_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

    with open(os.path.join(exp_dir, "DataSet.pkl"), "rb") as f:
        dataset = pickle.load(f)
    with open(os.path.join(exp_dir, "Conf.pkl"), "rb") as f:
        conf = pickle.load(f)
    with open(os.path.join(exp_dir, "Scores.pkl"), "rb") as f:
        scores = pickle.load(f)

    model_path = os.path.join(exp_dir, f"best_model_exp_{conf['ExpNumber']}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    model = conf["Model"]
    model.load_state_dict(torch.load(model_path, map_location=conf["Device"]))
    model.to(conf["Device"])
    model.eval()
    print(f"Loaded experiment {conf['ExpNumber']} from {exp_dir}")

    return dataset, conf, scores, model


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python 06_evaluate.py <exp_dir>")

    exp_dir = sys.argv[1]
    dataset, conf, scores, model = load_experiment(exp_dir)
    plots.plotting(dataset, model, conf, scores)
    print("Evaluation complete.")