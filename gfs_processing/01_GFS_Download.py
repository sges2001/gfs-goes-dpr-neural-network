"""
Download GFS numerical weather forecasts associated with DPR-GPM images.

For each image from the Dual-Frequency Precipitation Radar (DPR) aboard the
Global Precipitation Measurement (GPM) mission, this script identifies and
downloads the Global Forecast System (GFS) forecast closest in time to the
observation, from a model run initiated at least 6 hours beforehand.

GFS data source: https://rda.ucar.edu/datasets/d084001/
"""

import numpy as np
import pandas as pd
import os
import requests


def download_gfs_forecasts(df):
    """
    Download GFS GRIB2 files corresponding to each record in the DataFrame.

    For each DPR-GPM image, identifies the GFS model run initiated at least
    6 hours prior and the closest forecast lead time in 3-hour multiples.
    Files are saved to the 'Downloads/' folder, and a 'file_names' column
    is added to the DataFrame with the corresponding GFS filename.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least a 'file' column, where the filename
        encodes the DPR-GPM image timestamp at positions [17:32]
        in the format '%Y%m%d_%H%M%S'.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with the 'file_names' column added.
    """
    max_retries = 10  # If a download fails this many times, it is skipped
    destination_folder = "Downloads"
    os.makedirs(destination_folder, exist_ok=True)

    # Parse and process DPR-GPM image timestamps
    dates = pd.to_datetime(df["file"].str[17:32], format="%Y%m%d_%H%M%S")
    dates_rounded_3h = dates.dt.round(freq="3h")
    dates_minus_9h = dates - pd.Timedelta(hours=9)
    run_dates = dates_minus_9h.dt.round(freq="6h")  # GFS model run initiation time

    # Compute forecast lead time (in hours) as zero-padded 3-digit string
    lead_time = dates_rounded_3h - run_dates
    lead_time = lead_time.dt.components.hours
    lead_time = np.char.zfill(lead_time.values.astype(str), 3)

    date_no_hour = run_dates.dt.strftime("%Y%m%d")
    date_with_hour = run_dates.dt.strftime("%Y%m%d%H")
    year = run_dates.dt.strftime("%Y")

    # Build download URLs and output filenames
    urls = "https://data.rda.ucar.edu/d084001/" + year + "/" + date_no_hour + "/gfs.0p25." + date_with_hour + ".f" + lead_time + ".grib2"
    file_names = date_with_hour + "_f" + lead_time + ".grib2"
    df.loc[:, "file_names"] = file_names

    for i, (url, file_name) in enumerate(zip(urls, file_names)):
        print(f"Downloading file {i+1} of {len(df)} - {file_name}")
        full_path = os.path.join(destination_folder, file_name)

        if os.path.exists(full_path):
            print(f'File "{file_name}" already exists in the destination folder. Skipping.')
        else:
            attempts = 0
            while attempts < max_retries:
                try:
                    response = requests.get(url, timeout=120)  # Timeout to avoid hanging on unresponsive requests
                    if response.status_code == 200:
                        with open(full_path, "wb") as f:
                            f.write(response.content)
                        print(f"Download complete: {full_path}")
                        break  # Exit retry loop on success
                    else:
                        print(f"Failed to download {file_name} (attempt {attempts+1} of {max_retries}): HTTP {response.status_code}")
                except requests.exceptions.RequestException as e:
                    print(f"Error while downloading {file_name} (attempt {attempts+1} of {max_retries}): {e}")

                attempts += 1
                if attempts < max_retries:
                    print(f"Retrying {file_name}. Attempts remaining: {max_retries - attempts}")
                else:
                    print(f"Could not download {file_name} after {max_retries} attempts. Skipping.")

    return df


if __name__ == "__main__":
    data = pd.read_csv("metadata_files.csv")  # CSV with DPR-GPM image metadata
    data = download_gfs_forecasts(data)
    data.reset_index(drop=True, inplace=True)
    data.to_csv("metadata_files.csv", index=False)
