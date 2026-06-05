"""
Extraction and spatial subsetting of meteorological variables from GFS GRIB2 files.

For each record in the DPR-GPM metadata CSV, this script:
  - Opens the corresponding GFS GRIB2 file.
  - Extracts variables at isobaric levels (u, v, gh, t) at 850, 500, and 300 hPa.
  - Extracts surface and other level-type variables (prate, cape, cin, pwat, hlcy).
  - Subsets a ±5° domain centered on the DPR-GPM overpass location.
  - Saves the combined dataset as a compressed .npz file.
"""

import numpy as np
import pandas as pd
import os
import xarray as xr
import cfgrib


# Isobaric pressure levels to extract [hPa]
PRESSURE_LEVELS = [850.0, 500.0, 300.0]

# Spatial subset half-width in degrees
DELTA_DEGREES = 5


def file_already_exists(npz_filename, folder="processed_threshold"):
    """Check whether the .npz file has already been processed and saved."""
    return os.path.exists(os.path.join(folder, npz_filename))


def get_isobaric_var_records(datasets):
    """
    Scan a list of cfgrib datasets and identify variables
    associated with isobaric levels (isobaricInhPa).

    Parameters
    ----------
    datasets : list
        List of datasets returned by cfgrib.open_datasets().

    Returns
    -------
    varname_list : list of str
        Names of variables found at isobaric levels.
    varrecord_list : list of int
        Index of the dataset in the list to which each variable belongs.
    """
    varname_list = []
    varrecord_list = []
    for irec, ds in enumerate(datasets):
        if list(ds.coords)[2] == 'isobaricInhPa':
            for var in ds.keys():
                varname_list.append(var)
                varrecord_list.append(irec)
    return varname_list, varrecord_list


def compute_domain_indices(latitudes, longitudes, lat_center, lon_center_360, delta=DELTA_DEGREES):
    """
    Compute the array indices for a square domain
    centered on (lat_center, lon_center_360).

    Parameters
    ----------
    latitudes : np.ndarray
        Latitude array from the GRIB2 file.
    longitudes : np.ndarray
        Longitude array from the GRIB2 file (0–360° format).
    lat_center : float
        Central latitude of the subset domain.
    lon_center_360 : float
        Central longitude of the subset domain, in 0–360° format.
    delta : float
        Half-width of the domain in degrees.

    Returns
    -------
    ilat, elat, ilon, elon : int
        Minimum and maximum indices for latitude and longitude.
    """
    ind_lat = np.where(
        (latitudes >= lat_center - delta) & (latitudes <= lat_center + delta)
    )[0]
    ind_lon = np.where(
        (longitudes >= lon_center_360 - delta) & (longitudes <= lon_center_360 + delta)
    )[0]
    return ind_lat.min(), ind_lat.max(), ind_lon.min(), ind_lon.max()


def extract_isobaric_variables(ds_isobaric, varname_list, varrecord_list,
                                pressure_levels_array, ilat, elat, ilon, elon):
    """
    Extract u, v, gh, and t at the pressure levels defined in PRESSURE_LEVELS
    and return a single xr.Dataset with dimensions (pressure_level, latitude, longitude).

    Parameters
    ----------
    ds_isobaric : list
        List of cfgrib datasets containing isobaric data.
    varname_list : list of str
        Names of available isobaric variables.
    varrecord_list : list of int
        Dataset index corresponding to each variable.
    pressure_levels_array : np.ndarray
        Array of all pressure levels available in the file.
    ilat, elat, ilon, elon : int
        Spatial subset indices.

    Returns
    -------
    xr.Dataset
        Dataset containing u, v, gh, t at the selected pressure levels.
    """
    urec  = varrecord_list[varname_list.index('u')]
    vrec  = varrecord_list[varname_list.index('v')]
    ghrec = varrecord_list[varname_list.index('gh')]
    trec  = varrecord_list[varname_list.index('t')]

    lat = ds_isobaric[0].latitude.data[ilat:elat+1]
    lon = ds_isobaric[0].longitude.data[ilon:elon+1]

    datasets_per_level = []
    for level in PRESSURE_LEVELS:
        level_idx = np.where(pressure_levels_array == level)[0]
        if len(level_idx) == 0:
            print(f"  Warning: pressure level {level} hPa not found in file. Skipping.")
            continue

        ds_level = xr.Dataset(
            {
                "u":  (["latitude", "longitude"], np.squeeze(ds_isobaric[urec]['u'].values[level_idx,  ilat:elat+1, ilon:elon+1])),
                "v":  (["latitude", "longitude"], np.squeeze(ds_isobaric[vrec]['v'].values[level_idx,  ilat:elat+1, ilon:elon+1])),
                "gh": (["latitude", "longitude"], np.squeeze(ds_isobaric[ghrec]['gh'].values[level_idx, ilat:elat+1, ilon:elon+1])),
                "t":  (["latitude", "longitude"], np.squeeze(ds_isobaric[trec]['t'].values[level_idx,  ilat:elat+1, ilon:elon+1])),
            },
            coords={
                "latitude":       lat,
                "longitude":      lon,
                "pressure_level": [level],
            }
        )
        datasets_per_level.append(ds_level)

    return xr.concat(datasets_per_level, dim="pressure_level")


def open_surface_dataset(file_path):
    """
    Attempt to open surface-level variables from a GFS GRIB2 file.

    First tries to open the 'surface' level type directly. If that fails
    (because the file contains variables with mixed stepType values), it
    opens the file in two parts ('instant' and 'avg') and merges them.

    Parameters
    ----------
    file_path : str
        Path to the GRIB2 file.

    Returns
    -------
    xr.Dataset or None
        Surface dataset, or None if the file could not be opened.
    """
    try:
        return xr.open_dataset(file_path, engine='cfgrib',
                               filter_by_keys={'typeOfLevel': 'surface'})
    except Exception:
        pass  # File has mixed stepTypes; attempt to open in two parts

    try:
        ds_instant = xr.open_dataset(file_path, engine='cfgrib',
                                     filter_by_keys={'typeOfLevel': 'surface', 'stepType': 'instant'})
        ds_avg = xr.open_dataset(file_path, engine='cfgrib',
                                 filter_by_keys={'typeOfLevel': 'surface', 'stepType': 'avg'})
        # compat='override' resolves conflicts in duplicate variables (e.g. prate),
        # keeping the values from the first dataset (avg)
        return xr.merge([ds_avg, ds_instant], compat='override')
    except Exception as e:
        print(f"  Error opening surface data: {e}")
        return None


def process_grib_file(file_path, lat_center, lon_center):
    """
    Open a GFS GRIB2 file, extract variables of interest,
    and return an xr.Dataset subset to a ±5° domain.

    Parameters
    ----------
    file_path : str
        Path to the GRIB2 file.
    lat_center : float
        Central latitude of the DPR-GPM overpass.
    lon_center : float
        Central longitude of the DPR-GPM overpass (-180 to 180° format).

    Returns
    -------
    xr.Dataset or None
        Combined dataset with all variables, or None if an error occurred.
    """
    # GFS GRIB2 files store longitudes in 0–360° format
    lon_center_360 = (lon_center + 360) % 360

    # --- Isobaric block ---
    # cfgrib is used directly instead of xarray because xarray fails to recover
    # u and v wind components for forecast lead times beyond ~82000 s
    ds_isobaric = cfgrib.open_datasets(file_path)
    varname_list, varrecord_list = get_isobaric_var_records(ds_isobaric)
    pressure_levels_array = ds_isobaric[varrecord_list[0]].isobaricInhPa.data

    lat_all = ds_isobaric[0].latitude.data
    lon_all = ds_isobaric[0].longitude.data
    ilat, elat, ilon, elon = compute_domain_indices(lat_all, lon_all, lat_center, lon_center_360)

    ds_isobaric_subset = extract_isobaric_variables(
        ds_isobaric, varname_list, varrecord_list,
        pressure_levels_array, ilat, elat, ilon, elon
    )

    # Check for missing values in isobaric variables
    for var in ["u", "v", "gh", "t"]:
        if ds_isobaric_subset[var].isnull().any():
            print(f"  Warning: missing values found in '{var}' (isobaric).")

    # --- Surface level ---
    ds_surface = open_surface_dataset(file_path)
    if ds_surface is None:
        print("  Critical error: could not open surface level. Skipping file.")
        return None
    ds_surface = ds_surface[["prate", "cape", "cin"]]

    # --- atmosphereSingleLayer (precipitable water) ---
    ds_single_layer = xr.open_dataset(file_path, engine='cfgrib',
                                      filter_by_keys={'typeOfLevel': 'atmosphereSingleLayer'})
    ds_single_layer = ds_single_layer[["pwat"]]

    # --- heightAboveGroundLayer (storm relative helicity: hlcy) ---
    ds_3000m = xr.open_dataset(file_path, engine='cfgrib',
                                filter_by_keys={'typeOfLevel': 'heightAboveGroundLayer'})

    # Spatial subset for datasets opened with xarray
    ds_non_isobaric = xr.merge([ds_surface, ds_single_layer, ds_3000m])
    ds_non_isobaric = ds_non_isobaric.sel(
        latitude=slice(lat_center + DELTA_DEGREES, lat_center - DELTA_DEGREES),
        longitude=slice(lon_center_360 - DELTA_DEGREES, lon_center_360 + DELTA_DEGREES)
    )

    # Check for missing values in non-isobaric variables
    for var in ["prate", "cape", "cin", "hlcy"]:
        if ds_non_isobaric[var].isnull().any():
            print(f"  Warning: missing values found in '{var}'.")

    return xr.merge([ds_isobaric_subset, ds_non_isobaric])


if __name__ == "__main__":
    data = pd.read_csv("metadata_files.csv")
    output_folder = "processed_threshold"
    os.makedirs(output_folder, exist_ok=True)

    for i, row in data.iterrows():
        title     = f"{row['file_names']}_{row['lat']}_{row['lon']}"
        npz_filename = f"{title}.npz"

        if file_already_exists(npz_filename, output_folder):
            print(f"[{i+1}/{len(data)}] Already exists: {npz_filename}. Skipping.")
            continue

        print(f"[{i+1}/{len(data)}] Processing: {npz_filename}")
        full_path = os.path.join("Downloads", row['file_names'])

        ds_combined = process_grib_file(full_path, round(row["lat"], 3), round(row["lon"], 3))

        if ds_combined is None:
            continue

        # Store overpass metadata as dataset attributes
        ds_combined.attrs.update({
            'title':                  title,
            'central_latitude':       row['lat'],
            'central_longitude':      row['lon'],
            'dpr_filename':           row['file'],
            'gfs_run_and_lead_time':  row['file_names'],
        })

        # Save as compressed .npz
        data_dict = {var: ds_combined[var].values for var in ds_combined.data_vars}
        data_dict['attributes'] = ds_combined.attrs
        output_path = os.path.join(output_folder, npz_filename)
        np.savez_compressed(output_path, **data_dict)
        print(f"  Saved: {output_path}")