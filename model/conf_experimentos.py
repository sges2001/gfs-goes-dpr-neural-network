"""
Experiment configuration functions for the DPR-GPM precipitation retrieval model.

Each function defines the hyperparameters for a specific experiment and returns
them as a tuple. The naming convention is:

    exp_<loss_name>_<exp_number>()

where <loss_name> must match one of the loss families handled in main_train.py
(MSE, MSE_SF, MSE_inv_PDF, Quantile, MultiLoss) and <exp_number> is an integer
that distinguishes experiments within the same loss family.

Usage
-----
Called dynamically from main_train.py via:
    conf_experimentos.exp_<loss_name>_<exp_number>()
"""


def exp_MSE_1():
    """Baseline U-Net with MSE loss, LeakyReLU activations, [0,1] normalization."""
    ExpName, ExpNumber       = "MSE", 1
    ModelType                = "unet_sadeghi2020"
    ActType, OutActType      = "LeakyReLU", "LeakyReLU"
    init_dist, gain_function = "kaiming_normal", "leaky_relu"
    BatchSize, MaxEpochs     = 1024, 1000
    Norm, TypeNorm, Transform = True, "01", None
    LearningRate             = 1.0e-3
    scheduler_act            = True
    Milestones, Gamma        = [], 0.1
    WeightDecay, dropout_act = 0.0, False
    return (ExpName, ExpNumber, ModelType, ActType, OutActType,
            init_dist, gain_function, BatchSize, MaxEpochs,
            Norm, TypeNorm, Transform, LearningRate,
            scheduler_act, Milestones, Gamma, WeightDecay, dropout_act)


def exp_MSE_2():
    """Same as exp_MSE_1 but with standardized normalization and dropout enabled."""
    ExpName, ExpNumber       = "MSE", 2
    ModelType                = "unet_sadeghi2020"
    ActType, OutActType      = "LeakyReLU", "LeakyReLU"
    init_dist, gain_function = "kaiming_normal", "leaky_relu"
    BatchSize, MaxEpochs     = 1024, 1000
    Norm, TypeNorm, Transform = True, "standarized", None
    LearningRate             = 1.0e-3
    scheduler_act            = True
    Milestones, Gamma        = [300, 600], 0.1
    WeightDecay, dropout_act = 1.0e-5, True
    return (ExpName, ExpNumber, ModelType, ActType, OutActType,
            init_dist, gain_function, BatchSize, MaxEpochs,
            Norm, TypeNorm, Transform, LearningRate,
            scheduler_act, Milestones, Gamma, WeightDecay, dropout_act)


def exp_MSE_SF_1():
    """MSE with Softmax weighting (alpha=0.5), milestone-based LR scheduler."""
    ExpName, ExpNumber       = "MSE_SF", 1
    ModelType                = "unet_sadeghi2020"
    ActType, OutActType      = "LeakyReLU", "LeakyReLU"
    init_dist, gain_function = "kaiming_normal", "leaky_relu"
    BatchSize, MaxEpochs     = 1024, 1000
    Norm, TypeNorm, Transform = True, "01", None
    LearningRate             = 1.0e-3
    scheduler_act            = True
    Milestones, Gamma        = [300, 600], 0.1
    WeightDecay, dropout_act = 0.0, False
    alpha                    = 0.5
    return (ExpName, ExpNumber, ModelType, ActType, OutActType,
            init_dist, gain_function, BatchSize, MaxEpochs,
            Norm, TypeNorm, Transform, LearningRate,
            scheduler_act, Milestones, Gamma, WeightDecay, dropout_act, alpha)


def exp_MSE_inv_PDF_1():
    """MSE weighted by the inverse of the precipitation PDF (max_weight=10, factor=2)."""
    ExpName, ExpNumber       = "MSE_inv_PDF", 1
    ModelType                = "unet_sadeghi2020"
    ActType, OutActType      = "LeakyReLU", "LeakyReLU"
    init_dist, gain_function = "kaiming_normal", "leaky_relu"
    BatchSize, MaxEpochs     = 1024, 1000
    Norm, TypeNorm, Transform = True, "01", None
    LearningRate             = 1.0e-3
    scheduler_act            = True
    Milestones, Gamma        = [300, 600], 0.1
    WeightDecay, dropout_act = 0.0, False
    max_weight, weight_factor = 10, 2
    return (ExpName, ExpNumber, ModelType, ActType, OutActType,
            init_dist, gain_function, BatchSize, MaxEpochs,
            Norm, TypeNorm, Transform, LearningRate,
            scheduler_act, Milestones, Gamma, WeightDecay, dropout_act,
            max_weight, weight_factor)


def exp_Quantile_1():
    """Quantile loss targeting the 90th percentile (alpha=0.9)."""
    ExpName, ExpNumber       = "Quantile", 1
    ModelType                = "unet_sadeghi2020"
    ActType, OutActType      = "LeakyReLU", "LeakyReLU"
    init_dist, gain_function = "kaiming_normal", "leaky_relu"
    BatchSize, MaxEpochs     = 1024, 1000
    Norm, TypeNorm, Transform = True, "01", None
    LearningRate             = 1.0e-3
    scheduler_act            = True
    Milestones, Gamma        = [300, 600], 0.1
    WeightDecay, dropout_act = 0.0, False
    alpha                    = 0.9
    return (ExpName, ExpNumber, ModelType, ActType, OutActType,
            init_dist, gain_function, BatchSize, MaxEpochs,
            Norm, TypeNorm, Transform, LearningRate,
            scheduler_act, Milestones, Gamma, WeightDecay, dropout_act, alpha)


def exp_MultiLoss_1():
    """Combined multi-term loss, smaller batch size and lower initial LR."""
    ExpName, ExpNumber       = "MultiLoss", 1
    ModelType                = "unet_sadeghi2020"
    ActType, OutActType      = "LeakyReLU", "LeakyReLU"
    init_dist, gain_function = "kaiming_normal", "leaky_relu"
    BatchSize, MaxEpochs     = 512, 1000
    Norm, TypeNorm, Transform = True, "01", None
    LearningRate             = 5.0e-4
    scheduler_act            = True
    Milestones, Gamma        = [300, 600], 0.1
    WeightDecay, dropout_act = 1.0e-5, False
    return (ExpName, ExpNumber, ModelType, ActType, OutActType,
            init_dist, gain_function, BatchSize, MaxEpochs,
            Norm, TypeNorm, Transform, LearningRate,
            scheduler_act, Milestones, Gamma, WeightDecay, dropout_act)