# GOES-GFS-DPR Neural Network

Deep learning pipeline for precipitation rate retrieval over South America,
combining GOES-16 infrared brightness temperatures with GFS numerical weather
forecast fields. The model is trained against instantaneous precipitation
observations from the Dual-Frequency Precipitation Radar (DPR) aboard the
Global Precipitation Measurement (GPM) core satellite.


## Scope

This repository contains:

- GFS download and preprocessing pipeline.
- Dataset assembly and interpolation of GFS fields onto the GOES-16 grid.
- Neural network architecture, training, and evaluation code.

The preprocessing pipelines for GOES-16 imagery and GPM-DPR observations are
not included in this repository. The GFS processing workflow relies on metadata
generated from previously processed GOES-16 and GPM-DPR overpasses.

Therefore, this repository does not provide a complete end-to-end satellite
processing chain, but rather the GFS processing, dataset construction, and
model training components.

## Method

The retrieval is based on a dual-branch U-Net that processes two inputs in
parallel:

- **GOES-16** channel 13 (10.3 µm) brightness temperatures, resampled to a
  96×96 pixel domain centered on each GPM-DPR overpass.
- **GFS** 0.25° forecast fields interpolated onto the same GOES-16 grid.
  Available variables: precipitable water (pwat), CAPE, CIN, precipitation
  rate (prate), geopotential height, temperature, and wind components at
  850/500/300 hPa.

Each branch encodes its input independently. The feature maps are fused at
the bottleneck and decoded to a full-resolution precipitation field matching
the GOES-16 grid.

## Repository structure


```text
.
├── gfs_processing/
│   ├──metadata_files.csv        # Example of a CSV with DPR-GPM images metadata
│   ├── 01_GFS_Download.py       # Download GFS GRIB2 files from NCAR RDA
│   ├── 02_GFS_Extraction.py     # Extract variables and subset ±5° domain
│   ├── 03_GFS_Assembly.py       # Assemble per-overpass .npz into a dataset
│   └── 04_GFS_Interpolation.py  # Interpolate GFS fields onto the GOES-16 grid
│
└── model/
    ├── 05_main_train.py         # Training entry point
    ├── 06_evaluate.py           # Post-training evaluation and plots
    ├── conf_experimentos.py     # Experiment configuration functions
    ├── dataset.py               # Dataset construction, filtering, normalization
    ├── loss_functions.py        # Loss functions
    ├── models.py                # U-Net architectures and training loop
    ├── plots.py                # Diagnostic visualization
    ├── verification.py         # Verification metrics
    └── run_experiments.sh      # Batch launcher for multiple training runs
```
