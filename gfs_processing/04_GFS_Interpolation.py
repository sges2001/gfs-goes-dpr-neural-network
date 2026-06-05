"""
Interpolation of GFS forecast variables onto the GOES-16 grid.

For each DPR-GPM image, this script interpolates the GFS variables
(previously extracted and assembled in 03_GFS_Assembly.py) from the
GFS 0.25° grid onto the GOES-16 pixel grid associated with each overpass.
The interpolated fields are saved as compressed .npz files, one per variable.

Interpolation method: bilinear (scipy RegularGridInterpolator).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import pandas as pd


# --- Configuration ---
GFS_DATASET_PATH  = "assembled_dataset/gfs_dataset.npz"
DATAFRAME_PATH    = "metadata_files.csv"
GOES_DATA_DIR     = "path/to/goes_data"
OUTPUT_FOLDER     = "interpolated_data"
EXAMPLES_FOLDER   = os.path.join(OUTPUT_FOLDER, "examples")

OUTPUT_GRID_SIZE  = 96   # GOES-16 tile size (pixels)
N_PLOT_SAMPLES    = 10   # Number of randomly sampled images to plot


# --- Helper functions ---

def select_goes_tile(lat_goes, lon_goes, mid_lat):
    """
    Select the correct 96×96 tile from the GOES-16 image based on
    the central latitude of the DPR-GPM overpass.

    The GOES files contain two vertically stacked 96×96 tiles (192 rows total).
    The correct half is selected by comparing mid_lat against the latitude
    at the boundary row (row 96).

    Parameters
    ----------
    lat_goes : np.ndarray, shape (192, 96)
        Full latitude array from the GOES-16 file.
    lon_goes : np.ndarray, shape (192, 96)
        Full longitude array from the GOES-16 file.
    mid_lat : float
        Central latitude of the DPR-GPM overpass.

    Returns
    -------
    lat_tile : np.ndarray, shape (96, 96)
    lon_tile : np.ndarray, shape (96, 96)
    """
    boundary_lat = lat_goes[OUTPUT_GRID_SIZE, OUTPUT_GRID_SIZE // 2]
    increasing   = lat_goes[OUTPUT_GRID_SIZE + 1, OUTPUT_GRID_SIZE // 2] > boundary_lat

    if mid_lat > boundary_lat:
        if increasing:
            return lat_goes[OUTPUT_GRID_SIZE:, :], lon_goes[OUTPUT_GRID_SIZE:, :]
        else:
            return lat_goes[:OUTPUT_GRID_SIZE, :], lon_goes[:OUTPUT_GRID_SIZE, :]
    else:
        if increasing:
            return lat_goes[:OUTPUT_GRID_SIZE, :], lon_goes[:OUTPUT_GRID_SIZE, :]
        else:
            return lat_goes[OUTPUT_GRID_SIZE:, :], lon_goes[OUTPUT_GRID_SIZE:, :]


def interpolate_variable(var_name, var_data, gfs_lats, gfs_lons, dataframe, plot_indices):
    """
    Interpolate a GFS variable onto the GOES-16 grid for all DPR-GPM images.

    Parameters
    ----------
    var_name : str
        Variable name (used for filenames and plot labels).
    var_data : np.ndarray, shape (n_images, 40, 40)
        GFS field for all images.
    gfs_lats : np.ndarray, shape (n_images, 40)
        GFS latitude arrays for all images.
    gfs_lons : np.ndarray, shape (n_images, 40)
        GFS longitude arrays for all images.
    dataframe : pd.DataFrame
        DPR-GPM metadata, one row per image.
    plot_indices : set of int
        Image indices for which diagnostic plots will be saved.

    Returns
    -------
    np.ndarray, shape (n_images, OUTPUT_GRID_SIZE, OUTPUT_GRID_SIZE)
        Interpolated field on the GOES-16 grid.
    """
    n_images     = len(dataframe)
    interpolated = np.zeros((n_images, OUTPUT_GRID_SIZE, OUTPUT_GRID_SIZE), dtype=np.float32)

    prev_goes_path = None
    prev_mid_lat   = None

    print(f"Interpolating: {var_name}")

    for i, row in dataframe.iterrows():
        if i % 5000 == 0:
            print(f"  Processing image {i} / {n_images}")

        goes_path = os.path.join(GOES_DATA_DIR, row["file"])
        mid_lat   = row["lat"]

        # Consistency check: same GOES file should not be used for
        # images on opposite sides of the tile boundary
        if goes_path == prev_goes_path and prev_mid_lat is not None:
            ds_check     = xr.open_dataset(goes_path)
            boundary_lat = ds_check["Latitude"].values[OUTPUT_GRID_SIZE, OUTPUT_GRID_SIZE // 2]
            same_side    = (mid_lat > boundary_lat) == (prev_mid_lat > boundary_lat)
            if not same_side:
                print(f"  [WARNING] Image {i}: same GOES file used for images on opposite "
                      f"sides of the tile boundary ({prev_mid_lat:.2f}° vs {mid_lat:.2f}°). "
                      f"Skipping.")
                continue

        ds       = xr.open_dataset(goes_path)
        lat_goes = ds["Latitude"].values.copy()
        lon_goes = ds["Longitude"].values.copy()
        ds.close()

        lat_tile, lon_tile = select_goes_tile(lat_goes, lon_goes, mid_lat)

        lat_gfs = gfs_lats[i, :]
        lon_gfs = gfs_lons[i, :]
        field   = var_data[i, :, :]

        interpolator = RegularGridInterpolator(
            (lat_gfs, lon_gfs),
            field,
            method='linear',
            bounds_error=False,
            fill_value=None
        )
        points          = np.array([lat_tile.ravel(), lon_tile.ravel()]).T
        field_interp    = interpolator(points).reshape(lat_tile.shape)
        interpolated[i] = field_interp

        if i in plot_indices:
            save_diagnostic_plots(var_name, i, lon_gfs, lat_gfs, field,
                                  lon_tile, lat_tile, field_interp)

        prev_goes_path = goes_path
        prev_mid_lat   = mid_lat

    return interpolated


def save_diagnostic_plots(var_name, image_index, lon_orig, lat_orig, field_orig,
                           lon_interp, lat_interp, field_interp):
    """
    Save side-by-side diagnostic plots of the original GFS field and
    the interpolated field on the GOES-16 grid.

    Parameters
    ----------
    var_name : str
    image_index : int
    lon_orig, lat_orig : np.ndarray
        GFS grid coordinates.
    field_orig : np.ndarray, shape (40, 40)
        Original GFS field.
    lon_interp, lat_interp : np.ndarray
        GOES-16 tile coordinates.
    field_interp : np.ndarray, shape (96, 96)
        Interpolated field.
    """
    os.makedirs(EXAMPLES_FOLDER, exist_ok=True)
    vmin = np.nanmin(field_orig)
    vmax = np.nanmax(field_orig)
    label = var_name.upper()

    for suffix, lons, lats, field, title_tag in [
        ("original", lon_orig,  lat_orig,  field_orig,   "original (GFS grid)"),
        ("interp",   lon_interp, lat_interp, field_interp, "interpolated (GOES grid)"),
    ]:
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(lons, lats, field, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar(label=label)
        plt.title(f"{label} {title_tag} — image {image_index}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.savefig(os.path.join(EXAMPLES_FOLDER, f"{var_name}_{suffix}_{image_index}.png"), dpi=150)
        plt.close()


# --- Main ---

if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER,  exist_ok=True)
    os.makedirs(EXAMPLES_FOLDER, exist_ok=True)

    # Load GFS assembled dataset (output of 03_GFS_Assembly.py)
    gfs_dataset = np.load(GFS_DATASET_PATH, allow_pickle=True)
    gfs_lats    = gfs_dataset["latitudes"]   # shape (n_images, 40)
    gfs_lons    = gfs_dataset["longitudes"]  # shape (n_images, 40)

    dataframe = pd.read_csv(DATAFRAME_PATH)
    n_images  = len(dataframe)

    # Randomly sample images for diagnostic plots
    rng          = np.random.default_rng(seed=42)
    plot_indices = set(rng.choice(n_images, size=N_PLOT_SAMPLES, replace=False).tolist())

    # All GFS variables to interpolate
    variables = {
        "pwat": gfs_dataset["pwat"],
        "cape": gfs_dataset["cape"],
        "cin":  gfs_dataset["cin"],
        "prate":gfs_dataset["prate"],
        "u300": gfs_dataset["u300"],  "u500": gfs_dataset["u500"],  "u850": gfs_dataset["u850"],
        "v300": gfs_dataset["v300"],  "v500": gfs_dataset["v500"],  "v850": gfs_dataset["v850"],
        "t300": gfs_dataset["t300"],  "t500": gfs_dataset["t500"],  "t850": gfs_dataset["t850"],
        "gh300":gfs_dataset["gh300"], "gh500":gfs_dataset["gh500"], "gh850":gfs_dataset["gh850"],
    }

    for var_name, var_data in variables.items():
        interpolated = interpolate_variable(
            var_name, var_data, gfs_lats, gfs_lons, dataframe, plot_indices
        )
        out_path = os.path.join(OUTPUT_FOLDER, f"{var_name}_interpolated.npz")
        np.savez_compressed(out_path, **{var_name: interpolated})
        print(f"  Saved: {out_path}")