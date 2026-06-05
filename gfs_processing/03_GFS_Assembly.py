"""
Assembly of individual GFS .npz files into a single compressed dataset.

For each processed .npz file (one per DPR-GPM image), this script:
  - Reconstructs the lat/lon grid from the central coordinates stored as attributes.
  - Extracts isobaric variables (u, v, gh, t) at 850, 500, and 300 hPa.
  - Extracts surface and column-integrated variables (prate, cape, cin, pwat, hlcy).
  - Performs basic data quality checks (constant fields, binary-only fields, variables shapes).
  - Saves all variables into a single compressed .npz dataset.
"""

import os
import numpy as np
import pandas as pd


# --- Configuration ---

NPZ_FOLDER       = "processed_threshold"

DATAFRAME_PATH   = "metadata_files.csv"

OUTPUT_FOLDER    = "assembled_dataset"

GRID_SIZE        = 40       # Expected spatial dimensions (lat × lon)
GRID_RESOLUTION  = 0.25     # GFS grid spacing in degrees
DELTA_DEGREES    = 5        # Half-width of the spatial domain

PRESSURE_LEVELS  = [850, 500, 300]
ISOBARIC_VARS    = ("gh", "t", "u", "v")
SURFACE_VARS     = ("pwat", "cape", "cin", "prate", "hlcy")

# Isobaric variable names expanded by level: gh850, gh500, ..., v300
ISOBARIC_VAR_NAMES = [f"{var}{level}" for var in ISOBARIC_VARS for level in PRESSURE_LEVELS]


def reconstruct_latlon_grid(lat_center, lon_center):
    """
    Reconstruct the lat/lon grid for a ±DELTA_DEGREES domain centered
    on (lat_center, lon_center), aligned to the GFS 0.25° grid.

    Parameters
    ----------
    lat_center : float
        Central latitude of the DPR-GPM overpass.
    lon_center : float
        Central longitude of the DPR-GPM overpass.

    Returns
    -------
    lats : np.ndarray, shape (GRID_SIZE,)
    lons : np.ndarray, shape (GRID_SIZE,)
    """
    start_lat = np.floor((lat_center + DELTA_DEGREES) / GRID_RESOLUTION) * GRID_RESOLUTION
    end_lat   = np.ceil((lat_center  - DELTA_DEGREES) / GRID_RESOLUTION) * GRID_RESOLUTION
    lats = np.arange(start_lat, end_lat - 0.0001, -GRID_RESOLUTION)

    start_lon = np.ceil((lon_center  - DELTA_DEGREES) / GRID_RESOLUTION) * GRID_RESOLUTION
    end_lon   = np.floor((lon_center + DELTA_DEGREES) / GRID_RESOLUTION) * GRID_RESOLUTION
    lons = np.arange(start_lon, end_lon + 0.0001, GRID_RESOLUTION)

    # Trim to GRID_SIZE if rounding produced one extra point
    lats = lats[:GRID_SIZE]
    lons = lons[:GRID_SIZE]

    return lats, lons


def validate_grid_size(lats, lons, filename):
    """
    Verify that the reconstructed grid has the expected dimensions.
    Raises SystemExit if either dimension is incorrect.
    """
    if len(lats) != GRID_SIZE or len(lons) != GRID_SIZE:
        raise SystemExit(
            f"ERROR: unexpected grid size in '{filename}': "
            f"lats={len(lats)}, lons={len(lons)} (expected {GRID_SIZE})"
        )


def trim_to_grid(array_2d, filename, varname):
    """
    Trim a 2D array to (GRID_SIZE, GRID_SIZE) if it has one extra row or column.
    Raises SystemExit if the shape is not recoverable.
    """
    expected = (GRID_SIZE, GRID_SIZE)
    shape = array_2d.shape
    if shape == expected:
        return array_2d
    if shape in ((GRID_SIZE + 1, GRID_SIZE),
                 (GRID_SIZE, GRID_SIZE + 1),
                 (GRID_SIZE + 1, GRID_SIZE + 1)):
        return array_2d[:GRID_SIZE, :GRID_SIZE]
    raise SystemExit(
        f"ERROR: unrecoverable shape {shape} for '{varname}' in '{filename}'"
    )


def check_data_quality(array_2d, varname, filename):
    """
    Check whether a 2D field is spatially constant or contains only 0s and 1s.

    Returns
    -------
    str or None
        'constant', 'binary', or None if no issue was found.
    """
    if np.all(array_2d == array_2d.flat[0]):
        print(f"  [WARNING] '{varname}' in '{filename}' is spatially constant "
              f"(value: {array_2d.flat[0]:.4g})")
        return "constant"
    if np.all(np.isin(array_2d, [0, 1])):
        print(f"  [WARNING] '{varname}' in '{filename}' contains only 0s and 1s")
        return "binary"
    return None

def load_isobaric_variable(data, varname, filename):
    """
    Load an isobaric variable (shape 3×40×40) from an .npz file.
    Falls back to 'arr_0' and replicates it across 3 levels if the key is missing.

    Parameters
    ----------
    data : np.lib.npyio.NpzFile
    varname : str
        Variable name (e.g. 'u', 'gh').
    filename : str
        Used for warning messages.

    Returns
    -------
    np.ndarray, shape (3, GRID_SIZE, GRID_SIZE)
    """
    try:
        array = data[varname]
    except KeyError:
        print(f"  [WARNING] '{varname}' not found in '{filename}', falling back to 'arr_0'")
        array = data["arr_0"]
        array = np.stack([array] * len(PRESSURE_LEVELS), axis=0)

    # Trim spatial dimensions if needed
    return array[:, :GRID_SIZE, :GRID_SIZE]


def load_surface_variable(data, varname, filename):
    """
    Load a 2D surface variable (shape 40×40) from an .npz file.
    Falls back to 'arr_0' if the key is missing.

    Parameters
    ----------
    data : np.lib.npyio.NpzFile
    varname : str
    filename : str

    Returns
    -------
    np.ndarray, shape (GRID_SIZE, GRID_SIZE)
    """
    try:
        array = data[varname]
    except KeyError:
        print(f"  [WARNING] '{varname}' not found in '{filename}', falling back to 'arr_0'")
        try:
            array = data["arr_0"]
        except KeyError:
            raise SystemExit(
                f"ERROR: '{filename}' contains neither '{varname}' nor 'arr_0'"
            )
    return trim_to_grid(array, filename, varname)


# --- Main ---

if __name__ == "__main__":
    dataframe = pd.read_csv(DATAFRAME_PATH)
    dataframe = dataframe.head(1000)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    n_images = len(dataframe)

    # Pre-allocate output arrays
    isobaric_data = {var: np.zeros((n_images, GRID_SIZE, GRID_SIZE), dtype=float)
                     for var in ISOBARIC_VAR_NAMES}
    surface_data  = {var: np.zeros((n_images, GRID_SIZE, GRID_SIZE), dtype=float)
                     for var in SURFACE_VARS}
    data_lats     = np.zeros((n_images, GRID_SIZE), dtype=float)
    data_lons     = np.zeros((n_images, GRID_SIZE), dtype=float)

    # Quality-check trackers
    qc_constant_indices   = []
    qc_constant_variables = []
    qc_constant_values    = []
    qc_binary_indices     = []

    for i, row in dataframe.iterrows():
        filename = f"{row['file_names']}_{row['lat']}_{row['lon']}.npz"
        filepath = os.path.join(NPZ_FOLDER, filename)

        if not os.path.exists(filepath):
            print(f"  [WARNING] File not found: {filepath}. Skipping.")
            continue

        print(f"[{i+1}/{n_images}] Processing: {filename}")
        data = np.load(filepath, allow_pickle=True)

        # --- Recover central coordinates from attributes ---
        try:
            attrs       = data["attributes"].item()
            lat_center  = attrs.get("central_latitude",  0.0)
            lon_center  = attrs.get("central_longitude", 0.0)
        except KeyError:
            print(f"  [WARNING] No attributes found in '{filename}'. Using (0, 0).")
            lat_center, lon_center = 0.0, 0.0

        # --- Reconstruct and validate lat/lon grid ---
        lats, lons = reconstruct_latlon_grid(lat_center, lon_center)
        validate_grid_size(lats, lons, filename)
        data_lats[i] = lats
        data_lons[i] = lons

        # --- Isobaric variables ---
        for var in ISOBARIC_VARS:
            var_3d = load_isobaric_variable(data, var, filename)
            for j, level in enumerate(PRESSURE_LEVELS):
                key    = f"{var}{level}"
                slice_ = var_3d[j]
                isobaric_data[key][i] = slice_
                result = check_data_quality(slice_, key, filename)
                if result == "constant":
                    qc_constant_indices.append(i)
                    qc_constant_variables.append(key)
                    qc_constant_values.append(slice_.flat[0])
                elif result == "binary":
                    qc_binary_indices.append(i)
        # --- Surface and column-integrated variables ---
        for var in SURFACE_VARS:
            slice_ = load_surface_variable(data, var, filename)
            surface_data[var][i] = slice_
            result = check_data_quality(slice_, var, filename)
            if result == "constant":
                qc_constant_indices.append(i)
                qc_constant_variables.append(var)
                qc_constant_values.append(slice_.flat[0])
            elif result == "binary":
                qc_binary_indices.append(i)
        data.close()

    # --- Summary ---
    print("\n=== Isobaric variables ===")
    for var in sorted(isobaric_data):
        arr = isobaric_data[var]
        print(f"  {var}: shape={arr.shape}, "
              f"min={np.nanmin(arr):.4g}, max={np.nanmax(arr):.4g}, "
              f"mean={np.nanmean(arr):.4g}, NaNs={np.isnan(arr).sum()}")

    print("\n=== Surface variables ===")
    for var in SURFACE_VARS:
        arr = surface_data[var]
        print(f"  {var}: shape={arr.shape}, "
              f"min={np.nanmin(arr):.4g}, max={np.nanmax(arr):.4g}, "
              f"mean={np.nanmean(arr):.4g}, NaNs={np.isnan(arr).sum()}")

    # --- Save assembled dataset ---
    all_data = {**isobaric_data, **surface_data,
                "latitudes": data_lats, "longitudes": data_lons}
    np.savez_compressed(os.path.join(OUTPUT_FOLDER, "gfs_dataset.npz"), **all_data)
    print(f"\nDataset saved to: {os.path.join(OUTPUT_FOLDER, 'gfs_dataset.npz')}")

    # --- Save quality-check reports ---
    np.savez_compressed(
        os.path.join(OUTPUT_FOLDER, "qc_report.npz"),
        constant_indices   = np.array(qc_constant_indices),
        constant_variables = np.array(qc_constant_variables),
        constant_values    = np.array(qc_constant_values),
        binary_indices     = np.array(qc_binary_indices),
    )
    print("Quality check report saved.")