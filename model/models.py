"""
Neural network architectures for the DPR-GPM precipitation retrieval model.

All models take two inputs:
    - x_sat  : GOES-16 brightness temperature field, shape (B, H, W)
    - x_gfs  : GFS forecast field (downscaled), shape (B, H_gfs, W_gfs)

And produce a single-channel precipitation output, shape (B, H, W).

Available architectures:
    - unet_sadeghi2020          : Baseline dual-branch U-Net (32/64/128 channels).
    - unet_sadeghi2020_complex  : Wider variant (48/96/192 channels).
    - unet_sadeghi2020_many_jumps : Wider variant with skip connections from
                                    the GFS branch into both decoder stages.


"""

import os
import random
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import dataset as ds


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def define_seed(seed):
    """
    Set all random seeds for reproducibility across numpy, Python, and PyTorch.
    Also configures cuDNN to operate deterministically when a GPU is available.

    Parameters
    ----------
    seed : int
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark    = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(mode=True, warn_only=True)


# ---------------------------------------------------------------------------
# Weight initialization
# ---------------------------------------------------------------------------

def _initialize_weights(model, conf):
    """
    Initialize model weights according to the distribution specified in conf.

    Conv2d and Linear layers use Kaiming or Xavier initialization depending
    on conf['init_dist']. ConvTranspose2d uses Xavier normal. BatchNorm2d
    layers are initialized to weight=1, bias=0.

    Parameters
    ----------
    model : torch.nn.Module
    conf : dict
        Must contain 'init_dist' ('kaiming_normal' or 'kaiming_uniform')
        and 'gain_function' (e.g. 'leaky_relu').
    """
    gain = nn.init.calculate_gain("leaky_relu", 0.01)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if conf["init_dist"] == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity=conf["gain_function"])
            elif conf["init_dist"] == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.weight, mode="fan_out",
                                         nonlinearity=conf["gain_function"])
            if m.bias is not None:
                nn.init.normal_(m.bias, 0.0, 0.1)

        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_normal_(m.weight, gain)
            if m.bias is not None:
                nn.init.normal_(m.bias, 0.0, 0.1)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias,   0.0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain)
            nn.init.normal_(m.bias, 0.0, 0.1)


# ---------------------------------------------------------------------------
# Shared activation factory
# ---------------------------------------------------------------------------

def _build_activation(act_type, conf):
    """
    Instantiate a PyTorch activation function by name.
    For LeakyReLU, reads the negative_slope from conf (default 1e-2).

    Parameters
    ----------
    act_type : str or None
        Name of the activation class (e.g. 'LeakyReLU', 'ReLU').
        If None, returns None (no output activation).
    conf : dict

    Returns
    -------
    nn.Module or None
    """
    if act_type is None:
        return None
    if act_type == "LeakyReLU":
        slope = conf.get("negative_slope", 1e-2)
        return nn.LeakyReLU(negative_slope=slope)
    return getattr(nn, act_type)()


# ---------------------------------------------------------------------------
# Baseline dual-branch U-Net
# ---------------------------------------------------------------------------

class unet_sadeghi2020(nn.Module):
    """
    Dual-branch U-Net that fuses GOES-16 brightness temperatures with a
    downscaled GFS forecast field.

    Architecture
    ------------
    Satellite branch (encoder):
        Input (1, 96, 96) → conv1-2 → pool → conv3-4 → pool → conv5
        Skip connections saved after conv2 (x2) and conv4 (x4).

    GFS branch (encoder):
        Input (1, 6, 6) → conv1p-2p → pool → conv3p-4p → pool → upsample(12,12) → conv5p

    Fusion: concatenate satellite and GFS bottleneck features → conv5_bis

    Decoder:
        conv6 → upsample → cat(x4) → conv7-8 → upsample → cat(x2) → conv9-10 → conv11

    Parameters
    ----------
    conf : dict
        Must contain 'ActType', 'OutActType', and optionally 'negative_slope'.
    """
    def __init__(self, conf):
        super().__init__()

        # Satellite encoder
        self.conv1  = nn.Conv2d(1,   32,  3, padding=1, padding_mode="reflect")
        self.conv2  = nn.Conv2d(32,  32,  3, padding=1, padding_mode="reflect")
        self.conv3  = nn.Conv2d(32,  64,  3, padding=1, padding_mode="reflect")
        self.conv4  = nn.Conv2d(64,  64,  3, padding=1, padding_mode="reflect")
        self.conv5  = nn.Conv2d(64,  128, 3, padding=1, padding_mode="reflect")

        # GFS encoder
        self.conv1p = nn.Conv2d(1,   32,  3, padding=1, padding_mode="reflect")
        self.conv2p = nn.Conv2d(32,  32,  3, padding=1, padding_mode="reflect")
        self.conv3p = nn.Conv2d(32,  64,  3, padding=1, padding_mode="reflect")
        self.conv4p = nn.Conv2d(64,  64,  3, padding=1, padding_mode="reflect")
        self.conv5p = nn.Conv2d(64,  128, 3, padding=1, padding_mode="reflect")

        # Fusion + decoder
        self.conv5_bis = nn.Conv2d(256, 128, 3, padding=1, padding_mode="reflect")
        self.conv6     = nn.Conv2d(128, 64,  3, padding=1, padding_mode="reflect")
        self.conv7     = nn.Conv2d(128, 64,  3, padding=1, padding_mode="reflect")
        self.conv8     = nn.Conv2d(64,  32,  3, padding=1, padding_mode="reflect")
        self.conv9     = nn.Conv2d(64,  64,  3, padding=1, padding_mode="reflect")
        self.conv10    = nn.Conv2d(64,  32,  3, padding=1, padding_mode="reflect")
        self.conv11    = nn.Conv2d(32,  1,   3, padding=1, padding_mode="reflect")

        self.act_int = _build_activation(conf["ActType"],    conf)
        self.act_out = _build_activation(conf["OutActType"], conf)

        self.pool       = nn.MaxPool2d(2)
        self.up         = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # Upsample GFS bottleneck to match the satellite bottleneck spatial size (12×12)
        self.up_gfs     = nn.Upsample(size=(12, 12), mode="bilinear", align_corners=True)

    def forward(self, x_sat, x_gfs):
        # Satellite encoder
        x  = self.act_int(self.conv1(x_sat.unsqueeze(1)))
        x  = self.act_int(self.conv2(x))
        x2 = x.clone()                              # skip connection at 96×96
        x  = self.pool(x)
        x  = self.act_int(self.conv3(x))
        x  = self.act_int(self.conv4(x))
        x4 = x.clone()                              # skip connection at 48×48
        x  = self.pool(x)
        x  = self.act_int(self.conv5(x))            # bottleneck: 24×24 → 12×12

        # GFS encoder
        xp = self.act_int(self.conv1p(x_gfs.float().unsqueeze(1)))
        xp = self.act_int(self.conv2p(xp))
        xp = self.pool(xp)
        xp = self.act_int(self.conv3p(xp))
        xp = self.act_int(self.conv4p(xp))
        xp = self.pool(xp)
        xp = self.up_gfs(xp)                        # upsample to 12×12
        xp = self.act_int(self.conv5p(xp))

        # Fusion
        x = self.act_int(self.conv5_bis(torch.cat([x, xp], dim=1)))
        x = self.act_int(self.conv6(x))

        # Decoder
        x = self.up(x)
        x = torch.cat([x, x4], dim=1)
        x = self.act_int(self.conv7(x))
        x = self.act_int(self.conv8(x))
        x = self.up(x)
        x = torch.cat([x, x2], dim=1)
        x = self.act_int(self.conv9(x))
        x = self.act_int(self.conv10(x))
        x = self.act_out(self.conv11(x))

        return x.squeeze(1)


# ---------------------------------------------------------------------------
# Wider variant (48/96/192 channels)
# ---------------------------------------------------------------------------

class unet_sadeghi2020_complex(nn.Module):
    """
    Wider version of unet_sadeghi2020 with 48/96/192 channels instead of
    32/64/128. Same dual-branch U-Net topology. Increases model capacity
    at the cost of more parameters and memory.
    """
    def __init__(self, conf):
        super().__init__()

        self.conv1  = nn.Conv2d(1,   48,  3, padding=1, padding_mode="reflect")
        self.conv2  = nn.Conv2d(48,  48,  3, padding=1, padding_mode="reflect")
        self.conv3  = nn.Conv2d(48,  96,  3, padding=1, padding_mode="reflect")
        self.conv4  = nn.Conv2d(96,  96,  3, padding=1, padding_mode="reflect")
        self.conv5  = nn.Conv2d(96,  192, 3, padding=1, padding_mode="reflect")

        self.conv1p = nn.Conv2d(1,   48,  3, padding=1, padding_mode="reflect")
        self.conv2p = nn.Conv2d(48,  48,  3, padding=1, padding_mode="reflect")
        self.conv3p = nn.Conv2d(48,  96,  3, padding=1, padding_mode="reflect")
        self.conv4p = nn.Conv2d(96,  96,  3, padding=1, padding_mode="reflect")
        self.conv5p = nn.Conv2d(96,  192, 3, padding=1, padding_mode="reflect")

        self.conv5_bis = nn.Conv2d(384, 192, 3, padding=1, padding_mode="reflect")
        self.conv6     = nn.Conv2d(192, 96,  3, padding=1, padding_mode="reflect")
        self.conv7     = nn.Conv2d(192, 144, 3, padding=1, padding_mode="reflect")
        self.conv8     = nn.Conv2d(144, 48,  3, padding=1, padding_mode="reflect")
        self.conv9     = nn.Conv2d(96,  96,  3, padding=1, padding_mode="reflect")
        self.conv10    = nn.Conv2d(96,  48,  3, padding=1, padding_mode="reflect")
        self.conv11    = nn.Conv2d(48,  1,   3, padding=1, padding_mode="reflect")

        self.act_int = _build_activation(conf["ActType"],    conf)
        self.act_out = _build_activation(conf["OutActType"], conf)

        self.pool   = nn.MaxPool2d(2)
        self.up     = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_gfs = nn.Upsample(size=(12, 12),  mode="bilinear", align_corners=True)

    def forward(self, x_sat, x_gfs):
        x  = self.act_int(self.conv1(x_sat.unsqueeze(1)))
        x  = self.act_int(self.conv2(x))
        x2 = x.clone()
        x  = self.pool(x)
        x  = self.act_int(self.conv3(x))
        x  = self.act_int(self.conv4(x))
        x4 = x.clone()
        x  = self.pool(x)
        x  = self.act_int(self.conv5(x))

        xp = self.act_int(self.conv1p(x_gfs.float().unsqueeze(1)))
        xp = self.act_int(self.conv2p(xp))
        xp = self.pool(xp)
        xp = self.act_int(self.conv3p(xp))
        xp = self.act_int(self.conv4p(xp))
        xp = self.pool(xp)
        xp = self.up_gfs(xp)
        xp = self.act_int(self.conv5p(xp))

        x = self.act_int(self.conv5_bis(torch.cat([x, xp], dim=1)))
        x = self.act_int(self.conv6(x))
        x = self.up(x)
        x = torch.cat([x, x4], dim=1)
        x = self.act_int(self.conv7(x))
        x = self.act_int(self.conv8(x))
        x = self.up(x)
        x = torch.cat([x, x2], dim=1)
        x = self.act_int(self.conv9(x))
        x = self.act_int(self.conv10(x))
        x = self.act_out(self.conv11(x))

        return x.squeeze(1)


# ---------------------------------------------------------------------------
# Wider variant with GFS skip connections into both decoder stages
# ---------------------------------------------------------------------------

class unet_sadeghi2020_many_jumps(nn.Module):
    """
    Extension of unet_sadeghi2020_complex where intermediate GFS feature maps
    are also passed as skip connections into both decoder stages, not just
    at the bottleneck.

    Additional skip connections
    ---------------------------
    xp4 (after GFS conv4p, upsampled to 24×24) → concatenated with x4 at decoder stage 1
    xp2 (after GFS conv2p, upsampled to 48×48) → concatenated with x2 at decoder stage 2

    This allows the decoder to access GFS information at multiple spatial scales.
    """
    def __init__(self, conf):
        super().__init__()

        self.conv1  = nn.Conv2d(1,   48,  3, padding=1, padding_mode="reflect")
        self.conv2  = nn.Conv2d(48,  48,  3, padding=1, padding_mode="reflect")
        self.conv3  = nn.Conv2d(48,  96,  3, padding=1, padding_mode="reflect")
        self.conv4  = nn.Conv2d(96,  96,  3, padding=1, padding_mode="reflect")
        self.conv5  = nn.Conv2d(96,  192, 3, padding=1, padding_mode="reflect")

        self.conv1p = nn.Conv2d(1,   48,  3, padding=1, padding_mode="reflect")
        self.conv2p = nn.Conv2d(48,  48,  3, padding=1, padding_mode="reflect")
        self.conv3p = nn.Conv2d(48,  96,  3, padding=1, padding_mode="reflect")
        self.conv4p = nn.Conv2d(96,  96,  3, padding=1, padding_mode="reflect")
        self.conv5p = nn.Conv2d(96,  192, 3, padding=1, padding_mode="reflect")

        self.conv5_bis = nn.Conv2d(384, 192, 3, padding=1, padding_mode="reflect")
        self.conv6     = nn.Conv2d(192, 96,  3, padding=1, padding_mode="reflect")
        # Decoder stage 1: cat(x, x4, xp4) → 96+96+96 = 288 channels
        self.conv7     = nn.Conv2d(288, 192, 3, padding=1, padding_mode="reflect")
        self.conv8     = nn.Conv2d(192, 96,  3, padding=1, padding_mode="reflect")
        # Decoder stage 2: cat(x, x2, xp2) → 96+48+48 = 192 channels
        self.conv9     = nn.Conv2d(192, 96,  3, padding=1, padding_mode="reflect")
        self.conv10    = nn.Conv2d(96,  32,  3, padding=1, padding_mode="reflect")
        self.conv11    = nn.Conv2d(32,  1,   3, padding=1, padding_mode="reflect")

        self.act_int = _build_activation(conf["ActType"],    conf)
        self.act_out = _build_activation(conf["OutActType"], conf)

        self.pool      = nn.MaxPool2d(2)
        self.up        = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_gfs    = nn.Upsample(size=(12, 12), mode="bilinear", align_corners=True)
        self.up_gfs_24 = nn.Upsample(size=(24, 24), mode="bilinear", align_corners=True)
        self.up_gfs_48 = nn.Upsample(size=(48, 48), mode="bilinear", align_corners=True)

    def forward(self, x_sat, x_gfs):
        # Satellite encoder
        x  = self.act_int(self.conv1(x_sat.unsqueeze(1)))
        x  = self.act_int(self.conv2(x))
        x2 = x.clone()
        x  = self.pool(x)
        x  = self.act_int(self.conv3(x))
        x  = self.act_int(self.conv4(x))
        x4 = x.clone()
        x  = self.pool(x)
        x  = self.act_int(self.conv5(x))

        # GFS encoder — save intermediate maps for multi-scale skip connections
        xp  = self.act_int(self.conv1p(x_gfs.float().unsqueeze(1)))
        xp  = self.act_int(self.conv2p(xp))
        xp2 = xp.clone()                            # GFS skip at resolution 2
        xp  = self.pool(xp)
        xp  = self.act_int(self.conv3p(xp))
        xp  = self.act_int(self.conv4p(xp))
        xp4 = xp.clone()                            # GFS skip at resolution 4
        xp  = self.pool(xp)
        xp  = self.up_gfs(xp)
        xp  = self.act_int(self.conv5p(xp))

        # Fusion at bottleneck
        x = self.act_int(self.conv5_bis(torch.cat([x, xp], dim=1)))
        x = self.act_int(self.conv6(x))

        # Decoder stage 1: fuse with satellite and GFS skip at 24×24
        x   = self.up(x)
        xp4 = self.up_gfs_24(xp4)
        x   = torch.cat([x, x4, xp4], dim=1)
        x   = self.act_int(self.conv7(x))
        x   = self.act_int(self.conv8(x))

        # Decoder stage 2: fuse with satellite and GFS skip at 48×48
        x   = self.up(x)
        xp2 = self.up_gfs_48(xp2)
        x   = torch.cat([x, x2, xp2], dim=1)
        x   = self.act_int(self.conv9(x))
        x   = self.act_int(self.conv10(x))
        x   = self.act_out(self.conv11(x))

        return x.squeeze(1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def model_train(dataset, conf):
    """
    Train a model using the configuration in conf and return the trained model
    and a dictionary of training/validation scores.

    Training features:
        - Gradient clipping (max norm 1.0)
        - Optional early stopping with LR reduction (EarlyStoppingWithLRReduction)
        - Periodic model checkpoints at epochs in epocs_to_save
        - Per-epoch logging of MSE, variance, skewness loss components
        - Gradient norm diagnostics for the first 200 epochs

    Parameters
    ----------
    dataset : dict
        Output of ds.get_Data().
    conf : dict
        Training configuration (from 05_main_train.py).

    Returns
    -------
    model : torch.nn.Module
        Trained model (best checkpoint loaded).
    scores : dict
        Training and validation RMSE, bias, correlation, and loss per epoch.
    """
    model     = conf["Model"]
    optimizer = conf["Optimizer"](model.parameters(),
                                  lr=conf["LearningRate"],
                                  weight_decay=conf["WeightDecay"])
    loss_fn   = conf["Loss"]

    train_loader = DataLoader(dataset["TrainDataSet"],
                              batch_size=conf["BatchSize"],
                              shuffle=conf["Shuffle"])

    _initialize_weights(model, conf)
    model.to(conf["Device"])
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    early_stop = EarlyStoppingWithLRReduction(conf=conf, patience=20, verbose=True)

    # Epochs at which to save intermediate checkpoints
    epochs_to_save      = list(range(100, 501, 100))
    # Epochs for gradient norm diagnostics
    diagnostic_epochs   = set(range(200))

    scores = {k: [] for k in ["TrainLoss", "TrainRmse", "TrainBias", "TrainCorr",
                               "ValLoss",   "ValRmse",   "ValBias",   "ValCorr"]}

    grad_norms  = {"batch": [], "mse": [], "var": [], "skew": []}
    epoch_totals = {"mse": 0.0, "var": 0.0, "skew": 0.0}

    for epoch in range(conf["MaxEpochs"]):
        model.train()
        if hasattr(loss_fn, "set_epoch"):
            loss_fn.set_epoch(epoch)

        print(f"Epoch {epoch+1}/{conf['MaxEpochs']} — "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        batch_count = 0
        train_loss  = 0.0
        mse_epoch = var_epoch = skew_epoch = 0.0

        for x_sat, x_gfs, target in train_loader:
            optimizer.zero_grad()
            output = model(x_sat, x_gfs)
            total, mse, var, skew, kurt = loss_fn(output, target)

            # Gradient norm diagnostics for early epochs
            if epoch in diagnostic_epochs:
                for name, comp in [("batch", total), ("mse", mse),
                                   ("var", var), ("skew", skew)]:
                    if comp.grad_fn is not None:
                        grads = torch.autograd.grad(
                            comp, model.parameters(),
                            retain_graph=True, allow_unused=True
                        )
                        norm = torch.norm(
                            torch.stack([torch.norm(g, 2)
                                         for g in grads if g is not None]), 2
                        )
                        grad_norms[name].append(norm.item())

            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_count += 1
            train_loss  += total.item()
            mse_epoch   += mse.item()
            var_epoch   += var.item()
            skew_epoch  += skew.item()

        print(f"  Train loss: {train_loss/batch_count:.6f} | "
              f"MSE: {mse_epoch/batch_count:.6f} | "
              f"VAR: {var_epoch/batch_count:.6f} | "
              f"SKEW: {skew_epoch/batch_count:.6e}")

        epoch_totals["mse"]  += mse_epoch
        epoch_totals["var"]  += var_epoch
        epoch_totals["skew"] += skew_epoch

        # Evaluate on train and validation sets
        for split in ["Train", "Val"]:
            model.eval()
            _, _, target_eval, output_eval = model_eval(
                model, dataset[split + "DataSet"], conf["BatchSize"]
            )
            metrics = compute_scores(model, dataset, conf, split, loss_fn)

            loss_val = __loss__(output_eval, target_eval, loss_fn).cpu().detach().numpy()
            scores[split + "Loss"].append(loss_val)
            scores[split + "Rmse"].append(metrics["RMSE"])
            scores[split + "Bias"].append(metrics["BIAS"])
            scores[split + "Corr"].append(metrics["CORR"])

            if split == "Val":
                print(f"  Val loss: {loss_val:.6f} | "
                      f"RMSE: {metrics['RMSE']:.4f} | "
                      f"BIAS: {metrics['BIAS']:.4f} | "
                      f"CORR: {metrics['CORR']:.4f}")

            with torch.no_grad():
                torch.cuda.empty_cache()

        # Periodic checkpoint
        if epoch in epochs_to_save:
            ckpt_dir  = os.path.join(conf["OutPath"], "epoch_checkpoints", str(epoch))
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

        # Early stopping (grace period of 10 epochs)
        if epoch >= 10:
            if early_stop(scores["ValLoss"][-1], model, optimizer):
                if epoch < conf.get("MinEpochs", 0) - 1:
                    print(f"  Early stopping condition met at epoch {epoch}, "
                          f"but continuing until epoch {conf['MinEpochs']}.")
                else:
                    print(f"  Early stopping at epoch {epoch}.")
                    break

        with torch.no_grad():
            torch.cuda.empty_cache()
        gc.collect()

    # Summary of loss component averages
    n_epochs = max(epoch, 1)
    print(f"\nAverage MSE  over training: {epoch_totals['mse']  / n_epochs:.6f}")
    print(f"Average VAR  over training: {epoch_totals['var']  / n_epochs:.6f}")
    print(f"Average SKEW over training: {epoch_totals['skew'] / n_epochs:.6e}")

    # Summary of gradient norms
    for name, values in grad_norms.items():
        if values:
            print(f"Gradient norm [{name}]: "
                  f"min={np.min(values):.2e}  "
                  f"max={np.max(values):.2e}  "
                  f"mean={np.mean(values):.2e}")

    return model, scores


# ---------------------------------------------------------------------------
# Early stopping with learning rate reduction
# ---------------------------------------------------------------------------

class EarlyStoppingWithLRReduction:
    """
    Stop training if validation loss does not improve for `patience` epochs.
    Before stopping, reduces the learning rate up to `max_lr_reductions` times
    by a factor of `conf['Gamma']`, resetting the patience counter each time.

    The best model weights are saved to disk whenever the validation loss improves.

    Parameters
    ----------
    conf : dict
        Must contain 'Gamma', 'OutPath', 'ExpNumber'.
    patience : int
        Epochs to wait before reducing LR or stopping.
    delta : float
        Minimum improvement in validation loss to reset the counter.
    verbose : bool
    """
    def __init__(self, conf, patience=20, delta=0.0, verbose=False,
                 trace_func=print):
        self.patience          = patience
        self.delta             = delta
        self.verbose           = verbose
        self.gamma             = conf["Gamma"]
        self.checkpoint_path   = os.path.join(
            conf["OutPath"], f"best_model_exp_{conf['ExpNumber']}.pth"
        )
        self.trace_func        = trace_func
        self.best_metric       = float("inf")
        self.counter           = 0
        self.lr_reductions     = 0
        self.max_lr_reductions = 2

    def __call__(self, val_loss, model, optimizer):
        """
        Update state given the current validation loss.

        Returns True if training should stop, False otherwise.
        """
        if val_loss <= self.best_metric - self.delta:
            self.trace_func(
                f"  Validation loss improved "
                f"({self.best_metric:.6f} → {val_loss:.6f}). Saving model."
            )
            self.best_metric = val_loss
            torch.save(model.state_dict(), self.checkpoint_path)
            self.counter = 0
        else:
            self.counter += 1
            self.trace_func(
                f"  No improvement: {self.counter}/{self.patience} | "
                f"LR reductions: {self.lr_reductions}/{self.max_lr_reductions} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

        # Reduce LR if patience exceeded and reductions remain
        if self.counter >= self.patience and self.lr_reductions < self.max_lr_reductions:
            new_lr = optimizer.param_groups[0]["lr"] * self.gamma
            optimizer.param_groups[0]["lr"] = new_lr
            self.lr_reductions += 1
            self.counter = 0
            self.trace_func(
                f"  LR reduced to {new_lr:.2e} "
                f"({self.lr_reductions}/{self.max_lr_reductions})"
            )

        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def model_eval(model, dataset, batch_size, numpy=False):
    """
    Run inference on a dataset and return inputs, targets, and outputs.

    Parameters
    ----------
    model : torch.nn.Module
    dataset : torch.utils.data.Dataset
    batch_size : int
    numpy : bool
        If True, return numpy arrays instead of tensors.

    Returns
    -------
    input_sat, input_gfs, targets, outputs
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    sat_list, gfs_list, target_list, output_list = [], [], [], []
    with torch.no_grad():
        for x_sat, x_gfs, y in loader:
            sat_list.append(x_sat)
            gfs_list.append(x_gfs)
            target_list.append(y)
            output_list.append(model(x_sat, x_gfs))

    sat     = torch.cat(sat_list,    dim=0)
    gfs     = torch.cat(gfs_list,    dim=0)
    targets = torch.cat(target_list, dim=0)
    outputs = torch.cat(output_list, dim=0)

    if numpy:
        return (sat.cpu().numpy(), gfs.cpu().numpy(),
                targets.cpu().numpy(), outputs.cpu().detach().numpy())
    return sat, gfs, targets, outputs


def __rmse__(x, y):
    return torch.sqrt(torch.mean((x.flatten() - y.flatten()) ** 2)).cpu().numpy()

def __bias__(x, y):
    return torch.mean(x.flatten() - y.flatten()).cpu().numpy()

def __corr__(x, y):
    return np.corrcoef(x.flatten().cpu().numpy(), y.flatten().cpu().numpy())[0, 1]

def __loss__(output, target, loss_fn):
    """Compute loss with no gradient tracking. Returns the scalar total loss."""
    with torch.no_grad():
        result = loss_fn(output.clone().detach().float(),
                         target.clone().detach().float())
        return result[0] if isinstance(result, tuple) else result


def compute_scores(model, dataset, conf, split, loss_fn):
    """
    Evaluate the model on a given data split and return a metrics dictionary.

    Metrics are computed on denormalized values (physical units).
    RMSE and bias are also reported on the normalized scale (*275) for monitoring.

    Parameters
    ----------
    model : torch.nn.Module
    dataset : dict
    conf : dict
    split : str
        One of 'Train', 'Val', 'Test'.
    loss_fn : torch.nn.Module

    Returns
    -------
    dict with keys: RMSE, BIAS, CORR, Loss, Input_data_Sat,
                    Input_data_gfs, Target, Output
    """
    model.eval()
    sat, gfs, targets, outputs = model_eval(
        model, dataset[split + "DataSet"], conf["BatchSize"]
    )

    results = {
        "RMSE": __rmse__(outputs * 275, targets * 275),
        "BIAS": __bias__(outputs * 275, targets * 275),
        "CORR": __corr__(outputs, targets),
        "Loss": __loss__(outputs, targets, loss_fn),
    }

    sat, targets, outputs = ds.denorm(dataset, sat, targets, outputs)
    gfs, targets, outputs = ds.denormp(dataset, gfs, targets, outputs)

    results.update({
        "Input_data_Sat": sat,
        "Input_data_gfs": gfs,
        "Target":         targets,
        "Output":         outputs,
    })

    print(f"  [{split}] RMSE: {results['RMSE']:.4f} | "
          f"BIAS: {results['BIAS']:.4f} | CORR: {results['CORR']:.4f}")

    return results