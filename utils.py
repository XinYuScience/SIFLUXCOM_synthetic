import os
import psutil

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import geopandas as gpd
from shapely.vectorized import contains
import cartopy.io.shapereader as shpreader
from pathlib import Path
import xarray as xr
import random
from torch.utils.data import DataLoader
import time
import pandas as pd

PFT={'tropical':1,'temperate':2,'boreal':3}
# Helper Functions
# print memory usage
def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

# create a mask function: given 1d arrays of longitude and latitude, create a 2d boolean mask
def create_land_mask(ds):
    
    land_shp = shpreader.natural_earth(resolution='110m', category='physical', name='land')
    land_gdf = gpd.read_file(land_shp)
    land_polygon = land_gdf.unary_union

    # Extract the coordinate arrays (adjust these names if needed)
    lons = ds.longitude.values
    lats = ds.latitude.values

    # Create a meshgrid of coordinates (shape: [nlat, nlon])
    lon2d, lat2d = np.meshgrid(lons, lats)
    
    # Use vectorized contains to determine land pixels
    land_mask = contains(land_polygon, lon2d, lat2d)
    
    # Create an Antarctica mask (latitude < -60 degrees)
    antarctica_mask = lat2d < -60
    
    # Remove Antarctica from the land mask
    land_mask = land_mask & ~antarctica_mask

    return land_mask

def f_pft(ds):
    # Compute plant functional type ("pft") based on absolute latitude.
    ds["latitude_var"] = ds["latitude"].expand_dims(dim={"time": ds.dims["time"], "longitude": ds.dims["longitude"]}).transpose("time", "latitude", "longitude")
    
    ds["pft"] = xr.where(
        np.abs(ds["latitude_var"]) < 23.5, PFT["tropical"],
        np.where(np.abs(ds["latitude_var"]) < 66.5, PFT["temperate"], PFT["boreal"]))
    # ds["pft"] = xr.where(
    #     xr.ufuncs.fabs(ds["latitude_var"]) < 23.5, 1,
    #     xr.where(xr.ufuncs.fabs(ds["latitude_var"]) < 66.5, 2, 3))

    ds = ds.drop_vars("latitude_var")
    return ds

def f_ssrd(ssrd, pft):
    """
    Three distinct, roughly comparable functions for SSRD by PFT:
      1) Tropical => 1.2 * (ssrd + 5)^0.7
      2) Temperate => 10 + 3 * sqrt(ssrd + 2)
      3) Boreal => 10 * ln(ssrd + 20)
    """
    ssrd_arr = np.asarray(ssrd, dtype=float)
    pft_arr = np.asarray(pft, dtype=int)
    f_out = np.zeros_like(ssrd_arr)

    # 1) Tropical
    mask_trop = (pft_arr == PFT['tropical'])
    f_out[mask_trop] = 1.2 * (ssrd_arr[mask_trop] + 5) ** 0.7

    
    # 2) Temperate
    mask_temp = (pft_arr == PFT['temperate'])
    f_out[mask_temp] = 10.0 + 3.0 * np.sqrt(ssrd_arr[mask_temp] + 2)

    # 3) Boreal
    mask_boreal = (pft_arr == PFT['boreal'])
    f_out[mask_boreal] = 10.0 * np.log(ssrd_arr[mask_boreal] + 20)

    return f_out

def f_vpd_piecewise_decay(vpd, vpd_min=5.0, vpd_max=10.0, k=1.0):
    """
    Three-piece VPD-limiting function:
      1) f=1.0 when VPD <= vpd_min
      2) Exponential decay from 1 to 0 for vpd_min < VPD < vpd_max
      3) f=0.0 when VPD >= vpd_max
    
    The 'decay' portion is rescaled so that:
      f(vpd_min) = 1 and f(vpd_max) = 0 exactly.

    Parameters
    ----------
    vpd : float or np.array
          Vapor Pressure Deficit (kPa).
    vpd_min : float
          Below this VPD => f=1.
    vpd_max : float
          Above this VPD => f=0.
    k : float
          Exponential rate parameter. 
          Larger k => a steeper decay in [vpd_min, vpd_max].

    Returns
    -------
    f : float or np.array
        Values of the VPD-limiting function in [0,1].
    """
    # Convert input to numpy array for vectorized ops
    vpd_arr = np.asarray(vpd, dtype=float)
    f_out = np.zeros_like(vpd_arr)  # default: 0

    # 1) Region where VPD <= vpd_min => f = 1
    mask_low = (vpd_arr <= vpd_min)
    f_out[mask_low] = 1.0

    # 2) Region where vpd_min < VPD < vpd_max => exponential decay [1 -> 0]
    mask_mid = (vpd_arr > vpd_min) & (vpd_arr < vpd_max)
    # compute exponential at each point
    y_mid = np.exp(-k * (vpd_arr[mask_mid] - vpd_min))
    # exponential at vpd_max
    y_max = np.exp(-k * (vpd_max - vpd_min))
    # rescale so that f(vpd_min)=1 and f(vpd_max)=0
    # =>  f(VPD) = [ y(VPD) - y_max ] / [ 1 - y_max ]
    f_out[mask_mid] = (y_mid - y_max) / (1.0 - y_max)

    # 3) Region where VPD >= vpd_max => f=0 (already set)

    return f_out

def SIF_GPP_model_ssrd(ds, use_f_ssrd=False):
    # Constants for f_vpd_piecewise_decay function
    a = 1
    b = 0.1

    if use_f_ssrd:
        # Apply f_ssrd using properly broadcasted latitude
        ds = f_pft(ds)
        ds['ssrd_factor'] = (("time","latitude","longitude"),f_ssrd(ds["ssrd"], ds["pft"]))
        ds["gpp"] = ds['ssrd_factor'] * a
        ds["sif"] = ds['ssrd_factor'] * b
    else:
        ds["gpp"] = ds["ssrd"] * a
        ds["sif"] = ds["ssrd"] * b
    return ds

def SIF_GPP_model_ssrd_uncertainty(ds, gpp_uncertainty=0.01, sif_uncertainty=0.01):
    # Call the existing SIF_GPP_model to compute base values
    # ds = SIF_GPP_model_ssrd(ds)
    
    # Add noise: the standard deviation is proportional to the absolute value of the variable.
    ds["gpp"] = ds["gpp"] + np.random.normal(0, gpp_uncertainty * np.abs(ds["gpp"]), size=ds["gpp"].shape)
    ds["sif"] = ds["sif"] + np.random.normal(0, sif_uncertainty * np.abs(ds["sif"]), size=ds["sif"].shape)
    
    return ds
# SIF and GPP model to create the synthetic data
def SIF_GPP_model(ds, use_f_ssrd=False):
    # Constants for f_vpd_piecewise_decay function
    a = 1
    k1 = 1
    b = 0.1
    k2 = 0.5
    ds['vpd_factor_gpp'] = (("time","latitude","longitude"),f_vpd_piecewise_decay(ds["vpd"], k=k1))
    ds['vpd_factor_sif'] = (("time","latitude","longitude"),f_vpd_piecewise_decay(ds["vpd"], k=k2))

    if use_f_ssrd:
        # Apply f_ssrd using properly broadcasted latitude
        ds = f_pft(ds)
        ds['ssrd_factor'] = (("time","latitude","longitude"),f_ssrd(ds["ssrd"], ds["pft"]))
        ds["gpp"] = ds['ssrd_factor'] * a * ds['vpd_factor_gpp']
        ds["sif"] = ds['ssrd_factor'] * b * ds['vpd_factor_sif']
    else:
        ds["gpp"] = ds["ssrd"] * a * ds['vpd_factor_gpp']
        ds["sif"] = ds["ssrd"] * b * ds['vpd_factor_sif']
    return ds

# add normal uncertainty to the SIF and GPP output with uncertainty proportional to their magnitude
def SIF_GPP_model_uncertainty(ds, gpp_uncertainty=0.1, sif_uncertainty=0.1):
    # Call the existing SIF_GPP_model to compute base values
    ds = SIF_GPP_model(ds)
    
    # Add noise: the standard deviation is proportional to the absolute value of the variable.
    ds["gpp"] = ds["gpp"] + np.random.normal(0, gpp_uncertainty * np.abs(ds["gpp"]), size=ds["gpp"].shape)
    ds["sif"] = ds["sif"] + np.random.normal(0, sif_uncertainty * np.abs(ds["sif"]), size=ds["sif"].shape)
    
    return ds


# 1) calculate global mean and standard deviation for standardization
# 2) calculate the mean over time for each land pixel
# 3) randomly select 300 pixels that 150<SW_IN<200 (W/m2) and 5<VPD<10 (hPa)
def calculate_global_mean_std_pixel_mean(start_year=2012,end_year=2021,case='vpd3',use_f_ssrd=False):
    
    # Define the root directory
    root = Path("/Net/Groups/data_BGC/era5/e1/0d25_daily")
    # Define the paths to the data files
    ssrd_paths = sorted(
        (root / f"ssrd/ssrd.daily.fc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )
    vpd_paths = sorted(
        (root / f"vpd_cf/vpd_cf.daily.calc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )

    count = np.zeros(len(ssrd_paths))
    ssrd_mean_year = np.zeros(len(ssrd_paths))
    ssrd_var_year = np.zeros(len(ssrd_paths))
    vpd_mean_year = np.zeros(len(ssrd_paths))
    vpd_var_year = np.zeros(len(ssrd_paths))
    sif_mean_year = np.zeros(len(ssrd_paths))
    sif_var_year = np.zeros(len(ssrd_paths))
    gpp_mean_year = np.zeros(len(ssrd_paths))
    gpp_var_year = np.zeros(len(ssrd_paths))
    
    ds_yearly_means = []  # Accumulate each year's 2D mean map

    # ssrd_path = ssrd_paths[0]
    # vpd_path = vpd_paths[0]
    # i=0
    for i, (ssrd_path, vpd_path) in enumerate(zip(ssrd_paths, vpd_paths)):
        # Load one file at a time
        ds_ssrd = xr.open_dataset(ssrd_path)
        ds_ssrd = ds_ssrd * 11.574
        ds_vpd = xr.open_dataset(vpd_path)
        ds_vpd = ds_vpd.rename({"vpd_cf": "vpd"})
        ds = xr.merge([ds_ssrd, ds_vpd])

        # Calculate GPP and SIF
        ds = SIF_GPP_model_ssrd(ds,use_f_ssrd)
        # add uncertainty to the SIF and GPP
        ds = SIF_GPP_model_ssrd_uncertainty(ds)

        # land mask
        land_mask = create_land_mask(ds)
        ds = ds.where(land_mask)

        ###### pixel mean #####
        year_val = start_year + i
        # Calculate the yearly mean (2d map), expand with a 'year' dimension, and assign the year coordinate.
        if use_f_ssrd:
            ds_year = ds.drop_vars('pft').mean(dim="time", skipna=True)
        else:
            ds_year = ds.mean(dim="time", skipna=True)
        ds_year = ds_year.expand_dims({"year": [year_val]})
        ds_yearly_means.append(ds_year)
        #######################

        # Calculate the mean and sum of squares for the current year
        count[i] = np.sum(~np.isnan(ds["ssrd"].values))
        ssrd_mean_year[i] = (
            ds["ssrd"]
            .mean(dim=("time", "longitude", "latitude"), skipna=True)
            .values
        )
        ssrd_var_year[i] = (
            ds["ssrd"]
            .var(dim=("time", "longitude", "latitude"), skipna=True)
            .values
        )
        vpd_mean_year[i] = (
            ds["vpd"]
            .mean(dim=("time", "longitude", "latitude"), skipna=True)
            .values
        )
        vpd_var_year[i] = (
            ds["vpd"].var(dim=("time", "longitude", "latitude"), skipna=True).values
        )
        sif_mean_year[i] = (
            ds["sif"]
            .mean(dim=("time", "longitude", "latitude"), skipna=True)
            .values
        )
        sif_var_year[i] = (
            ds["sif"].var(dim=("time", "longitude", "latitude"), skipna=True).values
        )
        gpp_mean_year[i] = (
            ds["gpp"]
            .mean(dim=("time", "longitude", "latitude"), skipna=True)
            .values
        )
        gpp_var_year[i] = (
            ds["gpp"].var(dim=("time", "longitude", "latitude"), skipna=True).values
        )
    
    ######## pixel mean and select random pixels ########
    # Concatenate yearly maps along the new "year" dimension and calculate the overall mean.
    combined_years = xr.concat(ds_yearly_means, dim="year")
    overall_mean = combined_years.mean(dim="year", skipna=True) 
    
    sw_in_mean = overall_mean["ssrd"]
    vpd_mean = overall_mean["vpd"]

    # Identify valid pixels that fall within the specified ranges
    if use_f_ssrd:
        pft = ds['pft'][0,:,:]
        valid_mask = (sw_in_mean >= 150) & (sw_in_mean <= 200) & (vpd_mean >= 5) & (vpd_mean <= 10) & (pft == PFT['temperate'])
    else:
        valid_mask = (sw_in_mean >= 150) & (sw_in_mean <= 200) & (vpd_mean >= 5) & (vpd_mean <= 10)
    
    # Extract the indices of valid pixels
    valid_pixels = np.argwhere(valid_mask.values)
    # Randomly select 300 pixels if there are enough valid ones
    num_selected = min(300, len(valid_pixels))
    selected_indices = random.sample(list(valid_pixels), num_selected)

    #Save longitude and latitude of selected pixels.
    selected_indices_arr = np.array(selected_indices)
    # Assuming overall_mean has 1D latitude and longitude coordinates with dimensions "latitude" then "longitude".
    selected_lats = overall_mean["latitude"].values[selected_indices_arr[:,0]]
    selected_lons = overall_mean["longitude"].values[selected_indices_arr[:,1]]
    
    ###########################

    # Calculate the overall mean
    ssrd_mean_overall = np.sum(ssrd_mean_year * count) / np.sum(count)
    ssrd_std_overall = np.sqrt(
        (
            np.sum(count * ssrd_var_year)
            + np.sum(count * (ssrd_mean_year - ssrd_mean_overall) ** 2)
        )
        / np.sum(count)
    )
    vpd_mean_overall = np.sum(vpd_mean_year * count) / np.sum(count)
    vpd_std_overall = np.sqrt(
        (
            np.sum(count * vpd_var_year)
            + np.sum(count * (vpd_mean_year - vpd_mean_overall) ** 2)
        )
        / np.sum(count)
    )
    sif_mean_overall = np.sum(sif_mean_year * count) / np.sum(count)
    sif_std_overall = np.sqrt(
        (
            np.sum(count * sif_var_year)
            + np.sum(count * (sif_mean_year - sif_mean_overall) ** 2)
        )
        / np.sum(count)
    )
    gpp_mean_overall = np.sum(gpp_mean_year * count) / np.sum(count)
    gpp_std_overall = np.sqrt(
        (
            np.sum(count * gpp_var_year)
            + np.sum(count * (gpp_mean_year - gpp_mean_overall) ** 2)
        )
        / np.sum(count)
    )

    # Combine the results into a single dataset
    ds_combined_stats = xr.Dataset(
        {
            "SW_IN_mean": ssrd_mean_overall,
            "SW_IN_std": ssrd_std_overall,
            "VPD_mean": vpd_mean_overall,
            "VPD_std": vpd_std_overall,
            "SIF_mean": sif_mean_overall,
            "SIF_std": sif_std_overall,
            "GPP_mean": gpp_mean_overall,
            "GPP_std": gpp_std_overall,
        }
    )

    overall_mean.to_netcdf("./outputs/pixel_mean_"+case+".nc")
    np.save("./outputs/selected_latitudes_"+case+".npy", selected_lats)
    np.save("./outputs/selected_longitudes_"+case+".npy", selected_lons)
    ds_combined_stats.to_netcdf("./outputs/combined_stats_land_"+case+".nc")
    print_memory_usage()

# 1) calculate global mean and standard deviation for standardization
# 2) calculate the mean over time for each land pixel
def calculate_global_mean_std_pixel_mean_EVI(start_year=2012,end_year=2021,case='vpd3',use_f_ssrd=False):
    
    # Define the root directory
    root = Path("/Net/Groups/data_BGC/era5/e1/0d25_daily")
    # Define the paths to the data files
    ssrd_paths = sorted(
        (root / f"ssrd/ssrd.daily.fc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )
    vpd_paths = sorted(
        (root / f"vpd_cf/vpd_cf.daily.calc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )

    count = np.zeros(len(ssrd_paths))
    ssrd_mean_year = np.zeros(len(ssrd_paths))
    ssrd_var_year = np.zeros(len(ssrd_paths))
    vpd_mean_year = np.zeros(len(ssrd_paths))
    vpd_var_year = np.zeros(len(ssrd_paths))
    sif_mean_year = np.zeros(len(ssrd_paths))
    sif_var_year = np.zeros(len(ssrd_paths))
    gpp_mean_year = np.zeros(len(ssrd_paths))
    gpp_var_year = np.zeros(len(ssrd_paths))
    evi_mean_year = np.zeros(len(ssrd_paths))
    evi_var_year = np.zeros(len(ssrd_paths))
    
    ds_yearly_means = []  # Accumulate each year's 2D mean map

    # ssrd_path = ssrd_paths[0]
    # vpd_path = vpd_paths[0]
    # i=0
    for i, (ssrd_path, vpd_path) in enumerate(zip(ssrd_paths, vpd_paths)):
        # Load one file at a time
        ds_ssrd = xr.open_dataset(ssrd_path)
        ds_ssrd = ds_ssrd * 11.574
        ds_vpd = xr.open_dataset(vpd_path)
        ds_vpd = ds_vpd.rename({"vpd_cf": "vpd"})
        ds = xr.merge([ds_ssrd, ds_vpd])

        # Calculate GPP and SIF
        ds = SIF_GPP_EVI_model_ssrd(ds,use_f_ssrd)
        # add uncertainty to the SIF and GPP
        ds = SIF_GPP_EVI_model_ssrd_uncertainty(ds)

        # land mask
        land_mask = create_land_mask(ds)
        ds = ds.where(land_mask)

        ###### pixel mean #####
        year_val = start_year + i
        # Calculate the yearly mean (2d map), expand with a 'year' dimension, and assign the year coordinate.
        if use_f_ssrd:
            ds_year = ds.drop_vars('pft').mean(dim="time", skipna=True)
        else:
            ds_year = ds.mean(dim="time", skipna=True)
        ds_year = ds_year.expand_dims({"year": [year_val]})
        ds_yearly_means.append(ds_year)
        #######################

        # Calculate the mean and sum of squares for the current year
        count[i] = np.sum(~np.isnan(ds["ssrd"].values))
        ssrd_mean_year[i] = (
            ds["ssrd"]
            .mean(dim=("time", "longitude", "latitude"), skipna=True)
            .values
        )
        ssrd_var_year[i] = (
            ds["ssrd"]
            .var(dim=("time", "longitude", "latitude"), skipna=True)
            .values
        )
        vpd_mean_year[i] = (
            ds["vpd"]
            .mean(dim=("time", "longitude", "latitude"), skipna=True)
            .values
        )
        vpd_var_year[i] = (
            ds["vpd"].var(dim=("time", "longitude", "latitude"), skipna=True).values
        )
        sif_mean_year[i] = (
            ds["sif"]
            .mean(dim=("time", "longitude", "latitude"), skipna=True)
            .values
        )
        sif_var_year[i] = (
            ds["sif"].var(dim=("time", "longitude", "latitude"), skipna=True).values
        )
        gpp_mean_year[i] = (
            ds["gpp"]
            .mean(dim=("time", "longitude", "latitude"), skipna=True)
            .values
        )
        gpp_var_year[i] = (
            ds["gpp"].var(dim=("time", "longitude", "latitude"), skipna=True).values
        )
        evi_mean_year[i] = (
            ds["evi"]
            .mean(dim=("time", "longitude", "latitude"), skipna=True)
            .values
        )
        evi_var_year[i] = (
            ds["evi"].var(dim=("time", "longitude", "latitude"), skipna=True).values
        )
    
    ######## pixel mean and select random pixels ########
    # Concatenate yearly maps along the new "year" dimension and calculate the overall mean.
    combined_years = xr.concat(ds_yearly_means, dim="year")
    overall_mean = combined_years.mean(dim="year", skipna=True) 
    
    ###########################

    # Calculate the overall mean
    ssrd_mean_overall = np.sum(ssrd_mean_year * count) / np.sum(count)
    ssrd_std_overall = np.sqrt(
        (
            np.sum(count * ssrd_var_year)
            + np.sum(count * (ssrd_mean_year - ssrd_mean_overall) ** 2)
        )
        / np.sum(count)
    )
    vpd_mean_overall = np.sum(vpd_mean_year * count) / np.sum(count)
    vpd_std_overall = np.sqrt(
        (
            np.sum(count * vpd_var_year)
            + np.sum(count * (vpd_mean_year - vpd_mean_overall) ** 2)
        )
        / np.sum(count)
    )
    sif_mean_overall = np.sum(sif_mean_year * count) / np.sum(count)
    sif_std_overall = np.sqrt(
        (
            np.sum(count * sif_var_year)
            + np.sum(count * (sif_mean_year - sif_mean_overall) ** 2)
        )
        / np.sum(count)
    )
    gpp_mean_overall = np.sum(gpp_mean_year * count) / np.sum(count)
    gpp_std_overall = np.sqrt(
        (
            np.sum(count * gpp_var_year)
            + np.sum(count * (gpp_mean_year - gpp_mean_overall) ** 2)
        )
        / np.sum(count)
    )
    evi_mean_overall = np.sum(evi_mean_year * count) / np.sum(count)
    evi_std_overall = np.sqrt(
        (
            np.sum(count * evi_var_year)
            + np.sum(count * (evi_mean_year - evi_mean_overall) ** 2)
        )
        / np.sum(count)
    )

    # Combine the results into a single dataset
    ds_combined_stats = xr.Dataset(
        {
            "SW_IN_mean": ssrd_mean_overall,
            "SW_IN_std": ssrd_std_overall,
            "VPD_mean": vpd_mean_overall,
            "VPD_std": vpd_std_overall,
            "SIF_mean": sif_mean_overall,
            "SIF_std": sif_std_overall,
            "GPP_mean": gpp_mean_overall,
            "GPP_std": gpp_std_overall,
            "EVI_mean": evi_mean_overall,
            "EVI_std": evi_std_overall,
        }
    )

    overall_mean.to_netcdf("./outputs/pixel_mean_"+case+".nc")
    ds_combined_stats.to_netcdf("./outputs/combined_stats_land_"+case+".nc")
    print_memory_usage()

# 1) calculate global mean and standard deviation for standardization
# 2) calculate the mean over time for each land pixel
def calculate_global_mean_std_pixel_mean_EVI_encode(start_year=2012,end_year=2021,case='vpd3',use_f_ssrd=False):
    
    # Define the root directory
    root = Path("/Net/Groups/data_BGC/era5/e1/0d25_daily")
    # Define the paths to the data files
    ssrd_paths = sorted(
        (root / f"ssrd/ssrd.daily.fc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )
    vpd_paths = sorted(
        (root / f"vpd_cf/vpd_cf.daily.calc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )

    count = np.zeros(len(ssrd_paths))
    ssrd_mean_year = np.zeros(len(ssrd_paths))
    ssrd_var_year = np.zeros(len(ssrd_paths))
    vpd_mean_year = np.zeros(len(ssrd_paths))
    vpd_var_year = np.zeros(len(ssrd_paths))
    sif_mean_year = np.zeros(len(ssrd_paths))
    sif_var_year = np.zeros(len(ssrd_paths))
    gpp_mean_year = np.zeros(len(ssrd_paths))
    gpp_var_year = np.zeros(len(ssrd_paths))
    evi_mean_year = np.zeros(len(ssrd_paths))
    evi_var_year = np.zeros(len(ssrd_paths))
    
    ds_yearly_means = []  # Accumulate each year's 2D mean map

    # ssrd_path = ssrd_paths[0]
    # vpd_path = vpd_paths[0]
    # i=0
    for i, (ssrd_path, vpd_path) in enumerate(zip(ssrd_paths, vpd_paths)):
        # Load one file at a time
        ds_ssrd = xr.open_dataset(ssrd_path)
        ds_ssrd = ds_ssrd * 11.574
        ds_vpd = xr.open_dataset(vpd_path)
        ds_vpd = ds_vpd.rename({"vpd_cf": "vpd"})
        ds = xr.merge([ds_ssrd, ds_vpd])

        # Calculate GPP and SIF
        ds = SIF_GPP_EVI_model_ssrd(ds,use_f_ssrd)
        # add uncertainty to the SIF and GPP
        ds = SIF_GPP_EVI_model_ssrd_uncertainty(ds)

        # land mask
        land_mask = create_land_mask(ds)
        ds = ds.where(land_mask)

        ###### pixel mean #####
        year_val = start_year + i
        # Calculate the yearly mean (2d map), expand with a 'year' dimension, and assign the year coordinate.
        if use_f_ssrd:
            ds_year = ds.drop_vars('pft').mean(dim="time", skipna=True)
        else:
            ds_year = ds.mean(dim="time", skipna=True)
        ds_year = ds_year.expand_dims({"year": [year_val]})
        ds_yearly_means.append(ds_year)
        #######################

        # Calculate the mean and sum of squares for the current year
        count[i] = np.sum(~np.isnan(ds["ssrd"].values))
        ssrd_mean_year[i] = (
            ds["ssrd"]
            .mean(dim=("time", "longitude", "latitude"), skipna=True)
            .values
        )
        ssrd_var_year[i] = (
            ds["ssrd"]
            .var(dim=("time", "longitude", "latitude"), skipna=True)
            .values
        )
        vpd_mean_year[i] = (
            ds["vpd"]
            .mean(dim=("time", "longitude", "latitude"), skipna=True)
            .values
        )
        vpd_var_year[i] = (
            ds["vpd"].var(dim=("time", "longitude", "latitude"), skipna=True).values
        )
        sif_mean_year[i] = (
            ds["sif"]
            .mean(dim=("time", "longitude", "latitude"), skipna=True)
            .values
        )
        sif_var_year[i] = (
            ds["sif"].var(dim=("time", "longitude", "latitude"), skipna=True).values
        )
        gpp_mean_year[i] = (
            ds["gpp"]
            .mean(dim=("time", "longitude", "latitude"), skipna=True)
            .values
        )
        gpp_var_year[i] = (
            ds["gpp"].var(dim=("time", "longitude", "latitude"), skipna=True).values
        )
        evi_mean_year[i] = (
            ds["evi"]
            .mean(dim=("time", "longitude", "latitude"), skipna=True)
            .values
        )
        evi_var_year[i] = (
            ds["evi"].var(dim=("time", "longitude", "latitude"), skipna=True).values
        )
    
    ######## pixel mean and select random pixels ########
    # Concatenate yearly maps along the new "year" dimension and calculate the overall mean.
    combined_years = xr.concat(ds_yearly_means, dim="year")
    overall_mean = combined_years.mean(dim="year", skipna=True) 
    
    ###########################

    # Calculate the overall mean
    ssrd_mean_overall = np.sum(ssrd_mean_year * count) / np.sum(count)
    ssrd_std_overall = np.sqrt(
        (
            np.sum(count * ssrd_var_year)
            + np.sum(count * (ssrd_mean_year - ssrd_mean_overall) ** 2)
        )
        / np.sum(count)
    )
    vpd_mean_overall = np.sum(vpd_mean_year * count) / np.sum(count)
    vpd_std_overall = np.sqrt(
        (
            np.sum(count * vpd_var_year)
            + np.sum(count * (vpd_mean_year - vpd_mean_overall) ** 2)
        )
        / np.sum(count)
    )
    sif_mean_overall = np.sum(sif_mean_year * count) / np.sum(count)
    sif_std_overall = np.sqrt(
        (
            np.sum(count * sif_var_year)
            + np.sum(count * (sif_mean_year - sif_mean_overall) ** 2)
        )
        / np.sum(count)
    )
    gpp_mean_overall = np.sum(gpp_mean_year * count) / np.sum(count)
    gpp_std_overall = np.sqrt(
        (
            np.sum(count * gpp_var_year)
            + np.sum(count * (gpp_mean_year - gpp_mean_overall) ** 2)
        )
        / np.sum(count)
    )
    evi_mean_overall = np.sum(evi_mean_year * count) / np.sum(count)
    evi_std_overall = np.sqrt(
        (
            np.sum(count * evi_var_year)
            + np.sum(count * (evi_mean_year - evi_mean_overall) ** 2)
        )
        / np.sum(count)
    )

    # Combine the results into a single dataset
    ds_combined_stats = xr.Dataset(
        {
            "SW_IN_mean": ssrd_mean_overall,
            "SW_IN_std": ssrd_std_overall,
            "VPD_mean": vpd_mean_overall,
            "VPD_std": vpd_std_overall,
            "SIF_mean": sif_mean_overall,
            "SIF_std": sif_std_overall,
            "GPP_mean": gpp_mean_overall,
            "GPP_std": gpp_std_overall,
            "EVI_mean": evi_mean_overall,
            "EVI_std": evi_std_overall,
        }
    )

    overall_mean.to_netcdf("./outputs/pixel_mean_"+case+".nc")
    ds_combined_stats.to_netcdf("./outputs/combined_stats_land_"+case+".nc")
    print_memory_usage()

# Function to preprocess data for training
def preprocess_data(dataset, predictors, target):
    predictors_data = np.stack([dataset[var].values for var in predictors], axis=-1)  # Shape: (samples, features)
    predictors_data = predictors_data.reshape(-1, len(predictors))  # Flatten spatial dims
    target_data = dataset[target].values  # Shape: (samples,)
    target_data = target_data.reshape(-1)  # Flatten to match predictor shape

    valid_indices = np.all(~np.isnan(predictors_data), axis=1) & ~np.isnan(target_data)
    filtered_predictors = predictors_data[valid_indices]  # Shape: (valid_samples, features)
    filtered_target = target_data[valid_indices]  # Shape: (valid_samples,)

    return filtered_predictors, filtered_target


# Function to preprocess data for training
def preprocess_data_lat_lon(dataset, predictors, target):
    stacked_vars = np.stack([dataset[var].values for var in predictors], axis=-1)
    stacked_vars = stacked_vars.reshape(-1, len(predictors))
    target_data = dataset[target].values.reshape(-1)

    lat_vals = dataset.latitude.values
    lon_vals = dataset.longitude.values
    t_size = dataset.time.size if "time" in dataset.dims else 1
    time_idx, lat_idx, lon_idx = np.indices((t_size, lat_vals.size, lon_vals.size))

    lat_flat = lat_vals[lat_idx.ravel()]
    lon_flat = lon_vals[lon_idx.ravel()]
    time_flat = time_idx.ravel()

    valid = np.all(~np.isnan(stacked_vars), axis=1) & ~np.isnan(target_data)
    filtered_predictors = stacked_vars[valid]
    filtered_target = target_data[valid]
    filtered_lat = lat_flat[valid]
    filtered_lon = lon_flat[valid]
    filtered_time = time_flat[valid]

    return filtered_predictors, filtered_target, filtered_lat, filtered_lon, filtered_time

# function to pre training a neural network model
def pre_training(start_year=2017, end_year=2017,case='vpd3',use_f_ssrd=False):
    #read the global mean and std for standardization
    ds_stats = xr.open_dataset("./outputs/combined_stats_land_"+case+".nc")

    root = Path("/Net/Groups/data_BGC/era5/e1/0d25_daily")
    ssrd_paths = sorted(
        (root / f"ssrd/ssrd.daily.fc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )
    vpd_paths = sorted(
        (root / f"vpd_cf/vpd_cf.daily.calc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )

    # Initialize model, loss function, and optimizer
    input_dim = 2  # Number of predictors (ssrd and vpd)
    model = NeuralNet(input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model incrementally
    num_epochs = 50
    for ssrd_path, vpd_path in zip(ssrd_paths, vpd_paths):
        # Load one file at a time
        ds_ssrd = xr.open_dataset(ssrd_path)
        ds_ssrd = ds_ssrd * 11.574
        ds_vpd = xr.open_dataset(vpd_path)
        ds_vpd = ds_vpd.rename({"vpd_cf": "vpd"})
        ds = xr.merge([ds_ssrd, ds_vpd])

        # Calculate GPP and SIF
        ds = SIF_GPP_model(ds,use_f_ssrd)
        # add uncertainty to the SIF and GPP
        ds = SIF_GPP_model_uncertainty(ds)

        # land mask
        land_mask = create_land_mask(ds)
        ds = ds.where(land_mask)

        # Standardize ssrd, vpd, and sif using the pre-calculated global mean and std
        ds["ssrd"] = (ds["ssrd"] - ds_stats['SW_IN_mean'].values) / ds_stats['SW_IN_std'].values
        ds["vpd"] = (ds["vpd"] - ds_stats['VPD_mean'].values) / ds_stats['VPD_std'].values
        ds["sif"] = (ds["sif"] - ds_stats['SIF_mean'].values) / ds_stats['SIF_std'].values

        # Preprocess data
        predictors = ['ssrd', 'vpd']
        target_sif = 'sif'
        X_train_full, y_train_full = preprocess_data(ds, predictors, target_sif)
        
        # Split data into training and validation sets
        # X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train_full, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_full, dtype=torch.float32).unsqueeze(1)  # Ensure correct shape
        # X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        # y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

        # Create DataLoaders
        batch_size = 2**10  # You can adjust this value based on your available memory and GPU capacity
        train_loader = DataLoader(ClimateDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
        # val_loader = DataLoader(ClimateDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)

        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            # # Validation Step
            # model.eval()
            # total_val_loss = 0
            # with torch.no_grad():
            #     for batch_X, batch_y in val_loader:
            #         batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            #         outputs = model(batch_X)
            #         loss = criterion(outputs, batch_y)
            #         total_val_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            # avg_val_loss = total_val_loss / len(val_loader)

            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

            # Clear cache to free up memory
            torch.cuda.empty_cache()

        print_memory_usage()

    # Save the final model
    torch.save(model.state_dict(), "./outputs/model_weights_pre_training_"+case+".pth")
    print('pre training finished!')

def pre_training_forward_subsample(start_year=2012, end_year=2012,case='vpd3',
                                   sample_size=1000,use_f_ssrd=False,pft_code=True):
    # -------------------------------------------------------------------------
    # 1) Load global mean/std for standardization
    # -------------------------------------------------------------------------
    ds_stats = xr.open_dataset("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/combined_stats_land_"+case+".nc")
    # GPP_mean = ds_stats['GPP_mean'].values.item()
    # GPP_std  = ds_stats['GPP_std'].values.item()
    SIF_mean = ds_stats['SIF_mean'].values.item()
    SIF_std  = ds_stats['SIF_std'].values.item()

    SW_IN_mean = ds_stats['SW_IN_mean'].values.item()
    SW_IN_std  = ds_stats['SW_IN_std'].values.item()
    VPD_mean   = ds_stats['VPD_mean'].values.item()
    VPD_std    = ds_stats['VPD_std'].values.item()

    # -------------------------------------------------------------------------
    # 2) Load the selected pixel coordinates to *exclude* from evaluation
    # -------------------------------------------------------------------------
    selected_longitudes = np.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/selected_longitudes_"+case+".npy")
    selected_latitudes  = np.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/selected_latitudes_"+case+".npy")
    # selected_coords     = np.column_stack((selected_longitudes, selected_latitudes))

    # -------------------------------------------------------------------------
    # 3) Prepare ERA5 file paths for each year
    # -------------------------------------------------------------------------
    root = Path("/Net/Groups/data_BGC/era5/e1/0d25_daily")
    ssrd_paths = sorted(
        (root / f"ssrd/ssrd.daily.fc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )
    vpd_paths = sorted(
        (root / f"vpd_cf/vpd_cf.daily.calc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )

    """
    subsample the data to increase the training speed
    """
    ssrd_path = ssrd_paths[0]
    ds_temp = xr.open_dataset(ssrd_path)
    # # land mask
    # land_mask = create_land_mask(ds_temp)
    # ds_temp = ds_temp.where(land_mask)

    # #eddy covariance mask
    # EC_mask = subsampling(selected_coords,ds_temp)
    # ds_temp = ds_temp.where(~EC_mask)

    # valid_mask = ~np.isnan(ds_temp['ssrd'].isel(time=0))
    # valid_pixels = np.argwhere(valid_mask.values)
    # num_selected = min(sample_size, len(valid_pixels))
    # selected_indices = np.random.choice(valid_pixels.shape[0], num_selected, replace=False)
    # selected_positions = valid_pixels[selected_indices,:] 
    # #Save longitude and latitude of selected pixels.
    # # Assuming overall_mean has 1D latitude and longitude coordinates with dimensions "latitude" then "longitude".
    # selected_lats = ds_temp.latitude.values[selected_positions[:,0]]
    # selected_lons = ds_temp.longitude.values[selected_positions[:,1]]
    # sample_coords = np.column_stack((selected_lats, selected_lons))
    # mask_da = subsampling(sample_coords,ds_temp)
    mask_da = xr.open_dataarray("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/subsample_mask_forward_"+case+".nc")
    # mask_da.to_netcdf("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/subsample_mask_forward_"+case+".nc")
    # -------------------------------------------------------------------------
    # 4) Initialize the neural net and load weights
    # -------------------------------------------------------------------------
    if pft_code:
        predictors = ['ssrd', 'is_tropical', 'is_temperate', 'is_boreal']    
    else:
        predictors = ['ssrd']
    input_dim = len(predictors)  # we have 2 predictors: ssrd and vpd
    model = NeuralNet(input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if pft_code:
        model.load_state_dict(torch.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/model_weights_pre_training_"+case+"_one-hot_encoding.pth"))
    else:
        model.load_state_dict(torch.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/model_weights_pre_training_"+case+".pth"))
    model.eval()

    target_gpp = 'sif'

    # -------------------------------------------------------------------------
    # 5) Prepare large aggregator arrays for pixel-wise statistics
    # -------------------------------------------------------------------------
    # We won't know the *exact* shape until we open at least one file and check
    # the coordinate dimension lengths. For safety:
    lat_vals = ds_temp.latitude.values  # shape ~ (720,)
    lon_vals = ds_temp.longitude.values # shape ~ (1440,)
    
    # ssrd_path = ssrd_paths[0]
    # vpd_path = vpd_paths[0]
    years_range = np.arange(start_year, end_year + 1)
    for year_idx, ssrd_path, vpd_path in zip(years_range, ssrd_paths, vpd_paths):
        # ---- Load & rename
        ds_ssrd = xr.open_dataset(ssrd_path) * 11.574
        ds_vpd = xr.open_dataset(vpd_path).rename({"vpd_cf": "vpd"})
        ds = xr.merge([ds_ssrd, ds_vpd])

        # ---- Compute GPP and SIF, plus uncertainties
        ds = SIF_GPP_model(ds,use_f_ssrd)
        ds = SIF_GPP_model_uncertainty(ds)

        # ---- Standardize the input and the target
        ds["ssrd"] = (ds["ssrd"] - SW_IN_mean) / SW_IN_std
        ds["vpd"]  = (ds["vpd"]  - VPD_mean)   / VPD_std
        ds['sif']  = (ds['sif']  - SIF_mean)   / SIF_std

        if pft_code:
            ds['is_tropical'] = xr.where(ds['pft'] == PFT['tropical'], 1.0, 0.0)
            ds['is_temperate'] = xr.where(ds['pft'] == PFT['temperate'], 1.0, 0.0)
            ds['is_boreal'] = xr.where(ds['pft'] == PFT['boreal'], 1.0, 0.0)
        #subsample the data
        ds_remaining = ds.where(mask_da)

        # ---- Flatten data for PyTorch
        # We'll retrieve not just X_data, y_data but also the lat/lon indices.
        X_data, y_data, lat_idx_arr, lon_idx_arr, time_idx_arr = preprocess_data_lat_lon(
            ds_remaining, predictors, target_gpp
        )

        batch_size = 2**8
        dataset_climate = ClimateDataset_lat_lon(X_data, y_data, lat_idx_arr, lon_idx_arr, time_idx_arr)
        dataloader = DataLoader(dataset_climate, batch_size=batch_size, shuffle=False)

        pred_gpp_array = np.full((ds.dims["time"], ds.dims["latitude"], ds.dims["longitude"]), np.nan)

        with torch.no_grad():
            for batch_X, batch_y, lat_batch, lon_batch, time_batch in dataloader:
                outputs = model(batch_X.float().to(device))
                # Bring predictions and targets back to CPU
                outputs_np = outputs.cpu().numpy().squeeze()
                batch_y_np = batch_y.numpy()
                # Invert the standardization if you want metrics in original GPP units
                outputs_np_orig = outputs_np * SIF_std + SIF_mean
                batch_y_np_orig = batch_y_np * SIF_std + SIF_mean

                # Valid (non-NaN) mask
                valid_mask = ~np.isnan(outputs_np_orig) & ~np.isnan(batch_y_np_orig)
                outputs_valid = outputs_np_orig[valid_mask]
                lat_idx_valid = lat_batch[valid_mask]
                lon_idx_valid = lon_batch[valid_mask]
                time_idx_valid = time_batch[valid_mask]

                # Ensure lat_idx_valid and lon_idx_valid are numpy arrays
                if torch.is_tensor(lat_idx_valid):
                    lat_idx_valid = lat_idx_valid.cpu().numpy()
                if torch.is_tensor(lon_idx_valid):
                    lon_idx_valid = lon_idx_valid.cpu().numpy()

                lat_indices = np.array([np.where(lat_vals == val)[0][0] for val in lat_idx_valid])
                lon_indices = np.array([np.where(lon_vals == val)[0][0] for val in lon_idx_valid])

                # Assign predictions to the correct positions in pred_gpp_array
                pred_gpp_array[time_idx_valid, lat_indices, lon_indices] = outputs_valid

        ds_remaining['sif_hat'] = (("time","latitude","longitude"),pred_gpp_array)
        if pft_code:
            ds_remaining.to_netcdf("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/pre_training_forward_output_"+str(year_idx)+"_"+case+"_one-hot_encoding_subsample.nc")
        else:
            ds_remaining.to_netcdf("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/pre_training_forward_output_"+str(year_idx)+"_"+case+"_subsample.nc")
        # Clear cache
        torch.cuda.empty_cache()
        print(f"Processed year file: {ssrd_path.name}")
        print_memory_usage()

def pre_training_training_set(start_year=2017, end_year=2017,case='vpd3',
                                   sample_size=1000,use_f_ssrd=False,pft_code=True):
    # -------------------------------------------------------------------------
    # 1) Load global mean/std for standardization
    # -------------------------------------------------------------------------
    ds_stats = xr.open_dataset("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/combined_stats_land_"+case+".nc")
    # GPP_mean = ds_stats['GPP_mean'].values.item()
    # GPP_std  = ds_stats['GPP_std'].values.item()
    SIF_mean = ds_stats['SIF_mean'].values.item()
    SIF_std  = ds_stats['SIF_std'].values.item()

    SW_IN_mean = ds_stats['SW_IN_mean'].values.item()
    SW_IN_std  = ds_stats['SW_IN_std'].values.item()
    VPD_mean   = ds_stats['VPD_mean'].values.item()
    VPD_std    = ds_stats['VPD_std'].values.item()

    # -------------------------------------------------------------------------
    # 2) Load the selected pixel coordinates to *exclude* from evaluation
    # -------------------------------------------------------------------------
    selected_longitudes = np.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/selected_longitudes_"+case+".npy")
    selected_latitudes  = np.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/selected_latitudes_"+case+".npy")
    # selected_coords     = np.column_stack((selected_longitudes, selected_latitudes))

    # -------------------------------------------------------------------------
    # 3) Prepare ERA5 file paths for each year
    # -------------------------------------------------------------------------
    root = Path("/Net/Groups/data_BGC/era5/e1/0d25_daily")
    ssrd_paths = sorted(
        (root / f"ssrd/ssrd.daily.fc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )
    vpd_paths = sorted(
        (root / f"vpd_cf/vpd_cf.daily.calc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )

    """
    subsample the data to increase the training speed
    """
    ssrd_path = ssrd_paths[0]
    ds_temp = xr.open_dataset(ssrd_path)
    # # land mask
    # land_mask = create_land_mask(ds_temp)
    # ds_temp = ds_temp.where(land_mask)

    # #eddy covariance mask
    # EC_mask = subsampling(selected_coords,ds_temp)
    # ds_temp = ds_temp.where(~EC_mask)

    # valid_mask = ~np.isnan(ds_temp['ssrd'].isel(time=0))
    # valid_pixels = np.argwhere(valid_mask.values)
    # num_selected = min(sample_size, len(valid_pixels))
    # selected_indices = np.random.choice(valid_pixels.shape[0], num_selected, replace=False)
    # selected_positions = valid_pixels[selected_indices,:] 
    # #Save longitude and latitude of selected pixels.
    # # Assuming overall_mean has 1D latitude and longitude coordinates with dimensions "latitude" then "longitude".
    # selected_lats = ds_temp.latitude.values[selected_positions[:,0]]
    # selected_lons = ds_temp.longitude.values[selected_positions[:,1]]
    # sample_coords = np.column_stack((selected_lats, selected_lons))
    # mask_da = subsampling(sample_coords,ds_temp)
    mask_da = xr.open_dataarray("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/subsample_mask_pre_training_vpd3_pft.nc")
    # mask_da.to_netcdf("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/subsample_mask_forward_"+case+".nc")
    # -------------------------------------------------------------------------
    # 4) Initialize the neural net and load weights
    # -------------------------------------------------------------------------
    if pft_code:
        predictors = ['ssrd', 'is_tropical', 'is_temperate', 'is_boreal']    
    else:
        predictors = ['ssrd']
    input_dim = len(predictors)  # we have 2 predictors: ssrd and vpd
    model = NeuralNet_1024(input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if pft_code:
        model.load_state_dict(torch.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/model_weights_pre_training_"+case+"_one-hot_encoding_large.pth"))
    else:
        model.load_state_dict(torch.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/model_weights_pre_training_"+case+".pth"))
    model.eval()

    target_gpp = 'sif'

    # -------------------------------------------------------------------------
    # 5) Prepare large aggregator arrays for pixel-wise statistics
    # -------------------------------------------------------------------------
    # We won't know the *exact* shape until we open at least one file and check
    # the coordinate dimension lengths. For safety:
    lat_vals = ds_temp.latitude.values  # shape ~ (720,)
    lon_vals = ds_temp.longitude.values # shape ~ (1440,)
    
    # ssrd_path = ssrd_paths[0]
    # vpd_path = vpd_paths[0]
    years_range = np.arange(start_year, end_year + 1)
    for year_idx, ssrd_path, vpd_path in zip(years_range, ssrd_paths, vpd_paths):
        # ---- Load & rename
        ds_ssrd = xr.open_dataset(ssrd_path) * 11.574
        ds_vpd = xr.open_dataset(vpd_path).rename({"vpd_cf": "vpd"})
        ds = xr.merge([ds_ssrd, ds_vpd])

        # ---- Compute GPP and SIF, plus uncertainties
        ds = SIF_GPP_model_ssrd(ds,use_f_ssrd)
        ds = SIF_GPP_model_ssrd_uncertainty(ds)

        # ---- Standardize the input and the target
        ds["ssrd"] = (ds["ssrd"] - SW_IN_mean) / SW_IN_std
        ds["vpd"]  = (ds["vpd"]  - VPD_mean)   / VPD_std
        ds['sif']  = (ds['sif']  - SIF_mean)   / SIF_std
        
        #subsample the data
        ds = ds.where(mask_da)

        if pft_code:
            ds['is_tropical'] = xr.where(ds['pft'] == PFT['tropical'], 1.0, 0.0)
            ds['is_temperate'] = xr.where(ds['pft'] == PFT['temperate'], 1.0, 0.0)
            ds['is_boreal'] = xr.where(ds['pft'] == PFT['boreal'], 1.0, 0.0)

        # ---- Flatten data for PyTorch
        # We'll retrieve not just X_data, y_data but also the lat/lon indices.
        X_data, y_data, lat_idx_arr, lon_idx_arr, time_idx_arr = preprocess_data_lat_lon(
            ds, predictors, target_gpp
        )

        batch_size = 2**8
        dataset_climate = ClimateDataset_lat_lon(X_data, y_data, lat_idx_arr, lon_idx_arr, time_idx_arr)
        dataloader = DataLoader(dataset_climate, batch_size=batch_size, shuffle=False)

        pred_gpp_array = np.full((ds.dims["time"], ds.dims["latitude"], ds.dims["longitude"]), np.nan)

        with torch.no_grad():
            for batch_X, batch_y, lat_batch, lon_batch, time_batch in dataloader:
                outputs = model(batch_X.float().to(device))
                # Bring predictions and targets back to CPU
                outputs_np = outputs.cpu().numpy().squeeze()
                batch_y_np = batch_y.numpy()
                # Invert the standardization if you want metrics in original GPP units
                outputs_np_orig = outputs_np * SIF_std + SIF_mean
                batch_y_np_orig = batch_y_np * SIF_std + SIF_mean

                # Valid (non-NaN) mask
                valid_mask = ~np.isnan(outputs_np_orig) & ~np.isnan(batch_y_np_orig)
                outputs_valid = outputs_np_orig[valid_mask]
                lat_idx_valid = lat_batch[valid_mask]
                lon_idx_valid = lon_batch[valid_mask]
                time_idx_valid = time_batch[valid_mask]

                # Ensure lat_idx_valid and lon_idx_valid are numpy arrays
                if torch.is_tensor(lat_idx_valid):
                    lat_idx_valid = lat_idx_valid.cpu().numpy()
                if torch.is_tensor(lon_idx_valid):
                    lon_idx_valid = lon_idx_valid.cpu().numpy()

                lat_indices = np.array([np.where(lat_vals == val)[0][0] for val in lat_idx_valid])
                lon_indices = np.array([np.where(lon_vals == val)[0][0] for val in lon_idx_valid])

                # Assign predictions to the correct positions in pred_gpp_array
                pred_gpp_array[time_idx_valid, lat_indices, lon_indices] = outputs_valid

        ds['sif_hat'] = (("time","latitude","longitude"),pred_gpp_array)
        if pft_code:
            ds.to_netcdf("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/pre_training_forward_output_"+str(year_idx)+"_"+case+"_one-hot_encoding_training_set_subsample_large.nc")
        else:
            ds.to_netcdf("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/pre_training_forward_output_"+str(year_idx)+"_"+case+"_subsample.nc")
        # Clear cache
        torch.cuda.empty_cache()
        print(f"Processed year file: {ssrd_path.name}")
        print_memory_usage()

def pre_training_training_set_EVI(start_year=2017, end_year=2017,case='vpd3',
                                   sample_size=1000,use_f_ssrd=False,pft_code=True):
    # -------------------------------------------------------------------------
    # 1) Load global mean/std for standardization
    # -------------------------------------------------------------------------
    ds_stats = xr.open_dataset("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/combined_stats_land_"+case+".nc")
    # GPP_mean = ds_stats['GPP_mean'].values.item()
    # GPP_std  = ds_stats['GPP_std'].values.item()
    SIF_mean = ds_stats['SIF_mean'].values.item()
    SIF_std  = ds_stats['SIF_std'].values.item()

    SW_IN_mean = ds_stats['SW_IN_mean'].values.item()
    SW_IN_std  = ds_stats['SW_IN_std'].values.item()
    VPD_mean   = ds_stats['VPD_mean'].values.item()
    VPD_std    = ds_stats['VPD_std'].values.item()
    EVI_mean   = ds_stats['EVI_mean'].values.item()
    EVI_std    = ds_stats['EVI_std'].values.item()
    # -------------------------------------------------------------------------
    # 2) Load the selected pixel coordinates to *exclude* from evaluation
    # -------------------------------------------------------------------------
    # selected_longitudes = np.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/selected_longitudes_pft_best.npy")
    # selected_latitudes  = np.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/selected_latitudes_pft_best.npy")
    # selected_coords     = np.column_stack((selected_longitudes, selected_latitudes))

    # -------------------------------------------------------------------------
    # 3) Prepare ERA5 file paths for each year
    # -------------------------------------------------------------------------
    root = Path("/Net/Groups/data_BGC/era5/e1/0d25_daily")
    ssrd_paths = sorted(
        (root / f"ssrd/ssrd.daily.fc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )
    vpd_paths = sorted(
        (root / f"vpd_cf/vpd_cf.daily.calc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )

    """
    subsample the data to increase the training speed
    """
    ssrd_path = ssrd_paths[0]
    ds_temp = xr.open_dataset(ssrd_path)
    # # land mask
    # land_mask = create_land_mask(ds_temp)
    # ds_temp = ds_temp.where(land_mask)

    # #eddy covariance mask
    # EC_mask = subsampling(selected_coords,ds_temp)
    # ds_temp = ds_temp.where(~EC_mask)

    # valid_mask = ~np.isnan(ds_temp['ssrd'].isel(time=0))
    # valid_pixels = np.argwhere(valid_mask.values)
    # num_selected = min(sample_size, len(valid_pixels))
    # selected_indices = np.random.choice(valid_pixels.shape[0], num_selected, replace=False)
    # selected_positions = valid_pixels[selected_indices,:] 
    # #Save longitude and latitude of selected pixels.
    # # Assuming overall_mean has 1D latitude and longitude coordinates with dimensions "latitude" then "longitude".
    # selected_lats = ds_temp.latitude.values[selected_positions[:,0]]
    # selected_lons = ds_temp.longitude.values[selected_positions[:,1]]
    # sample_coords = np.column_stack((selected_lats, selected_lons))
    # mask_da = subsampling(sample_coords,ds_temp)
    mask_da = xr.open_dataarray("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/subsample_mask_pre_training_vpd3_pft.nc")
    # mask_da.to_netcdf("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/subsample_mask_forward_"+case+".nc")
    # -------------------------------------------------------------------------
    # 4) Initialize the neural net and load weights
    # -------------------------------------------------------------------------
    if pft_code:
        predictors = ['ssrd', 'evi']    
    else:
        predictors = ['ssrd']
    input_dim = len(predictors)  # we have 2 predictors: ssrd and vpd
    model = NeuralNet(input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if pft_code:
        model.load_state_dict(torch.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/model_weights_pre_training_"+case+".pth"))
    else:
        model.load_state_dict(torch.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/model_weights_pre_training_"+case+".pth"))
    model.eval()

    target_gpp = 'sif'

    # -------------------------------------------------------------------------
    # 5) Prepare large aggregator arrays for pixel-wise statistics
    # -------------------------------------------------------------------------
    # We won't know the *exact* shape until we open at least one file and check
    # the coordinate dimension lengths. For safety:
    lat_vals = ds_temp.latitude.values  # shape ~ (720,)
    lon_vals = ds_temp.longitude.values # shape ~ (1440,)
    
    # ssrd_path = ssrd_paths[0]
    # vpd_path = vpd_paths[0]
    years_range = np.arange(start_year, end_year + 1)
    for year_idx, ssrd_path, vpd_path in zip(years_range, ssrd_paths, vpd_paths):
        # ---- Load & rename
        ds_ssrd = xr.open_dataset(ssrd_path) * 11.574
        ds_vpd = xr.open_dataset(vpd_path).rename({"vpd_cf": "vpd"})
        ds = xr.merge([ds_ssrd, ds_vpd])

        # ---- Compute GPP and SIF, plus uncertainties
        ds = SIF_GPP_EVI_model_ssrd(ds,use_f_ssrd)
        ds = SIF_GPP_EVI_model_ssrd_uncertainty(ds)

        # ---- Standardize the input and the target
        ds["ssrd"] = (ds["ssrd"] - SW_IN_mean) / SW_IN_std
        ds["vpd"]  = (ds["vpd"]  - VPD_mean)   / VPD_std
        ds['sif']  = (ds['sif']  - SIF_mean)   / SIF_std
        ds['evi']  = (ds['evi']  - EVI_mean)   / EVI_std

        #subsample the data
        ds = ds.where(mask_da)

        # if pft_code:
        #     ds['is_tropical'] = xr.where(ds['pft'] == PFT['tropical'], 1.0, 0.0)
        #     ds['is_temperate'] = xr.where(ds['pft'] == PFT['temperate'], 1.0, 0.0)
        #     ds['is_boreal'] = xr.where(ds['pft'] == PFT['boreal'], 1.0, 0.0)

        # ---- Flatten data for PyTorch
        # We'll retrieve not just X_data, y_data but also the lat/lon indices.
        X_data, y_data, lat_idx_arr, lon_idx_arr, time_idx_arr = preprocess_data_lat_lon(
            ds, predictors, target_gpp
        )

        batch_size = 2**8
        dataset_climate = ClimateDataset_lat_lon(X_data, y_data, lat_idx_arr, lon_idx_arr, time_idx_arr)
        dataloader = DataLoader(dataset_climate, batch_size=batch_size, shuffle=False)

        pred_gpp_array = np.full((ds.dims["time"], ds.dims["latitude"], ds.dims["longitude"]), np.nan)

        with torch.no_grad():
            for batch_X, batch_y, lat_batch, lon_batch, time_batch in dataloader:
                outputs = model(batch_X.float().to(device))
                # Bring predictions and targets back to CPU
                outputs_np = outputs.cpu().numpy().squeeze()
                batch_y_np = batch_y.numpy()
                # Invert the standardization if you want metrics in original GPP units
                outputs_np_orig = outputs_np * SIF_std + SIF_mean
                batch_y_np_orig = batch_y_np * SIF_std + SIF_mean

                # Valid (non-NaN) mask
                valid_mask = ~np.isnan(outputs_np_orig) & ~np.isnan(batch_y_np_orig)
                outputs_valid = outputs_np_orig[valid_mask]
                lat_idx_valid = lat_batch[valid_mask]
                lon_idx_valid = lon_batch[valid_mask]
                time_idx_valid = time_batch[valid_mask]

                # Ensure lat_idx_valid and lon_idx_valid are numpy arrays
                if torch.is_tensor(lat_idx_valid):
                    lat_idx_valid = lat_idx_valid.cpu().numpy()
                if torch.is_tensor(lon_idx_valid):
                    lon_idx_valid = lon_idx_valid.cpu().numpy()

                lat_indices = np.array([np.where(lat_vals == val)[0][0] for val in lat_idx_valid])
                lon_indices = np.array([np.where(lon_vals == val)[0][0] for val in lon_idx_valid])

                # Assign predictions to the correct positions in pred_gpp_array
                pred_gpp_array[time_idx_valid, lat_indices, lon_indices] = outputs_valid

        ds['sif_hat'] = (("time","latitude","longitude"),pred_gpp_array)
        if pft_code:
            ds.to_netcdf("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/pre_training_forward_output_"+str(year_idx)+"_"+case+"_subsample.nc")
        else:
            ds.to_netcdf("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/pre_training_forward_output_"+str(year_idx)+"_"+case+"_subsample.nc")
        # Clear cache
        torch.cuda.empty_cache()
        print(f"Processed year file: {ssrd_path.name}")
        print_memory_usage()

def pre_training_training_set_EVI_pft_mean(start_year=2017, end_year=2017,case='vpd3',
                                   sample_size=1000,use_f_ssrd=False,pft_code=True):
    # -------------------------------------------------------------------------
    # 1) Load global mean/std for standardization
    # -------------------------------------------------------------------------
    ds_stats = xr.open_dataset("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/combined_stats_land_pft_EVI.nc")
    # GPP_mean = ds_stats['GPP_mean'].values.item()
    # GPP_std  = ds_stats['GPP_std'].values.item()
    SIF_mean = ds_stats['SIF_mean'].values.item()
    SIF_std  = ds_stats['SIF_std'].values.item()

    SW_IN_mean = ds_stats['SW_IN_mean'].values.item()
    SW_IN_std  = ds_stats['SW_IN_std'].values.item()
    VPD_mean   = ds_stats['VPD_mean'].values.item()
    VPD_std    = ds_stats['VPD_std'].values.item()
    EVI_mean   = ds_stats['EVI_mean'].values.item()
    EVI_std    = ds_stats['EVI_std'].values.item()
    # -------------------------------------------------------------------------
    # 2) Load the selected pixel coordinates to *exclude* from evaluation
    # -------------------------------------------------------------------------
    # selected_longitudes = np.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/selected_longitudes_pft_best.npy")
    # selected_latitudes  = np.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/selected_latitudes_pft_best.npy")
    # selected_coords     = np.column_stack((selected_longitudes, selected_latitudes))

    # -------------------------------------------------------------------------
    # 3) Prepare ERA5 file paths for each year
    # -------------------------------------------------------------------------
    root = Path("/Net/Groups/data_BGC/era5/e1/0d25_daily")
    ssrd_paths = sorted(
        (root / f"ssrd/ssrd.daily.fc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )
    vpd_paths = sorted(
        (root / f"vpd_cf/vpd_cf.daily.calc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )

    """
    subsample the data to increase the training speed
    """
    ssrd_path = ssrd_paths[0]
    ds_temp = xr.open_dataset(ssrd_path)
    # # land mask
    # land_mask = create_land_mask(ds_temp)
    # ds_temp = ds_temp.where(land_mask)

    # #eddy covariance mask
    # EC_mask = subsampling(selected_coords,ds_temp)
    # ds_temp = ds_temp.where(~EC_mask)

    # valid_mask = ~np.isnan(ds_temp['ssrd'].isel(time=0))
    # valid_pixels = np.argwhere(valid_mask.values)
    # num_selected = min(sample_size, len(valid_pixels))
    # selected_indices = np.random.choice(valid_pixels.shape[0], num_selected, replace=False)
    # selected_positions = valid_pixels[selected_indices,:] 
    # #Save longitude and latitude of selected pixels.
    # # Assuming overall_mean has 1D latitude and longitude coordinates with dimensions "latitude" then "longitude".
    # selected_lats = ds_temp.latitude.values[selected_positions[:,0]]
    # selected_lons = ds_temp.longitude.values[selected_positions[:,1]]
    # sample_coords = np.column_stack((selected_lats, selected_lons))
    # mask_da = subsampling(sample_coords,ds_temp)
    mask_da = xr.open_dataarray("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/subsample_mask_pre_training_vpd3_pft.nc")
    # mask_da.to_netcdf("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/subsample_mask_forward_"+case+".nc")
    # -------------------------------------------------------------------------
    # 4) Initialize the neural net and load weights
    # -------------------------------------------------------------------------
    if pft_code:
        predictors = ['ssrd', 'evi']    
    else:
        predictors = ['ssrd']
    input_dim = len(predictors)  # we have 2 predictors: ssrd and vpd
    model = NeuralNet(input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if pft_code:
        model.load_state_dict(torch.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/model_weights_pre_training_"+case+".pth"))
    else:
        model.load_state_dict(torch.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/model_weights_pre_training_"+case+".pth"))
    model.eval()

    target_gpp = 'sif'

    # -------------------------------------------------------------------------
    # 5) Prepare large aggregator arrays for pixel-wise statistics
    # -------------------------------------------------------------------------
    # We won't know the *exact* shape until we open at least one file and check
    # the coordinate dimension lengths. For safety:
    lat_vals = ds_temp.latitude.values  # shape ~ (720,)
    lon_vals = ds_temp.longitude.values # shape ~ (1440,)
    
    # ssrd_path = ssrd_paths[0]
    # vpd_path = vpd_paths[0]
    years_range = np.arange(start_year, end_year + 1)
    for year_idx, ssrd_path, vpd_path in zip(years_range, ssrd_paths, vpd_paths):
        # ---- Load & rename
        ds_ssrd = xr.open_dataset(ssrd_path) * 11.574
        ds_vpd = xr.open_dataset(vpd_path).rename({"vpd_cf": "vpd"})
        ds = xr.merge([ds_ssrd, ds_vpd])

        # ---- Compute GPP and SIF, plus uncertainties
        ds = SIF_GPP_EVI_model_ssrd(ds,use_f_ssrd)
        ds = SIF_GPP_EVI_model_ssrd_uncertainty(ds)

        # Compute the tropical EVI time series (average over longitude and latitude)
        evi_tropical_series = ds['evi'].where(ds['pft'] == PFT['tropical']).mean(dim=['longitude', 'latitude'])
        evi_tropical = evi_tropical_series.broadcast_like(ds['evi']).where(ds['pft'] == PFT['tropical'])
        # Compute the temperate EVI time series similarly
        evi_temperate_series = ds['evi'].where(ds['pft'] == PFT['temperate']).mean(dim=['longitude', 'latitude'])
        evi_temperate = evi_temperate_series.broadcast_like(ds['evi']).where(ds['pft'] == PFT['temperate'])
        # Compute the boreal EVI time series similarly
        evi_boreal_series = ds['evi'].where(ds['pft'] == PFT['boreal']).mean(dim=['longitude', 'latitude'])
        evi_boreal = evi_boreal_series.broadcast_like(ds['evi']).where(ds['pft'] == PFT['boreal'])
        # Combine the three EVI arrays along a new 'pft' dimension
        evi_merged = evi_tropical.combine_first(evi_temperate).combine_first(evi_boreal)
        
        ds['evi'] = evi_merged

        # ---- Standardize the input and the target
        ds["ssrd"] = (ds["ssrd"] - SW_IN_mean) / SW_IN_std
        # ds["vpd"]  = (ds["vpd"]  - VPD_mean)   / VPD_std
        ds['sif']  = (ds['sif']  - SIF_mean)   / SIF_std
        ds['evi']  = (ds['evi']  - EVI_mean)   / EVI_std

        #subsample the data
        ds = ds.where(mask_da)

        # if pft_code:
        #     ds['is_tropical'] = xr.where(ds['pft'] == PFT['tropical'], 1.0, 0.0)
        #     ds['is_temperate'] = xr.where(ds['pft'] == PFT['temperate'], 1.0, 0.0)
        #     ds['is_boreal'] = xr.where(ds['pft'] == PFT['boreal'], 1.0, 0.0)

        # ---- Flatten data for PyTorch
        # We'll retrieve not just X_data, y_data but also the lat/lon indices.
        X_data, y_data, lat_idx_arr, lon_idx_arr, time_idx_arr = preprocess_data_lat_lon(
            ds, predictors, target_gpp
        )

        batch_size = 2**8
        dataset_climate = ClimateDataset_lat_lon(X_data, y_data, lat_idx_arr, lon_idx_arr, time_idx_arr)
        dataloader = DataLoader(dataset_climate, batch_size=batch_size, shuffle=False)

        pred_gpp_array = np.full((ds.dims["time"], ds.dims["latitude"], ds.dims["longitude"]), np.nan)

        with torch.no_grad():
            for batch_X, batch_y, lat_batch, lon_batch, time_batch in dataloader:
                outputs = model(batch_X.float().to(device))
                # Bring predictions and targets back to CPU
                outputs_np = outputs.cpu().numpy().squeeze()
                batch_y_np = batch_y.numpy()
                # Invert the standardization if you want metrics in original GPP units
                outputs_np_orig = outputs_np * SIF_std + SIF_mean
                batch_y_np_orig = batch_y_np * SIF_std + SIF_mean

                # Valid (non-NaN) mask
                valid_mask = ~np.isnan(outputs_np_orig) & ~np.isnan(batch_y_np_orig)
                outputs_valid = outputs_np_orig[valid_mask]
                lat_idx_valid = lat_batch[valid_mask]
                lon_idx_valid = lon_batch[valid_mask]
                time_idx_valid = time_batch[valid_mask]

                # Ensure lat_idx_valid and lon_idx_valid are numpy arrays
                if torch.is_tensor(lat_idx_valid):
                    lat_idx_valid = lat_idx_valid.cpu().numpy()
                if torch.is_tensor(lon_idx_valid):
                    lon_idx_valid = lon_idx_valid.cpu().numpy()

                lat_indices = np.array([np.where(lat_vals == val)[0][0] for val in lat_idx_valid])
                lon_indices = np.array([np.where(lon_vals == val)[0][0] for val in lon_idx_valid])

                # Assign predictions to the correct positions in pred_gpp_array
                pred_gpp_array[time_idx_valid, lat_indices, lon_indices] = outputs_valid

        ds['sif_hat'] = (("time","latitude","longitude"),pred_gpp_array)
        if pft_code:
            ds.to_netcdf("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/pre_training_forward_output_"+str(year_idx)+"_"+case+"_subsample.nc")
        else:
            ds.to_netcdf("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/pre_training_forward_output_"+str(year_idx)+"_"+case+"_subsample.nc")
        # Clear cache
        torch.cuda.empty_cache()
        print(f"Processed year file: {ssrd_path.name}")
        print_memory_usage()

def pre_training_training_set_EVI_encode(start_year=2017, end_year=2017,case='vpd3',
                                   sample_size=1000,use_f_ssrd=False,pft_code=True):
    
    # -------------------------------------------------------------------------
    # 1) Load global mean/std for standardization
    # -------------------------------------------------------------------------
    ds_stats = xr.open_dataset("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/combined_stats_land_pft_EVI.nc")
    # GPP_mean = ds_stats['GPP_mean'].values.item()
    # GPP_std  = ds_stats['GPP_std'].values.item()
    SIF_mean = ds_stats['SIF_mean'].values.item()
    SIF_std  = ds_stats['SIF_std'].values.item()

    SW_IN_mean = ds_stats['SW_IN_mean'].values.item()
    SW_IN_std  = ds_stats['SW_IN_std'].values.item()
    VPD_mean   = ds_stats['VPD_mean'].values.item()
    VPD_std    = ds_stats['VPD_std'].values.item()
    EVI_mean   = ds_stats['EVI_mean'].values.item()
    EVI_std    = ds_stats['EVI_std'].values.item()
    # -------------------------------------------------------------------------
    # 2) Load the selected pixel coordinates to *exclude* from evaluation
    # -------------------------------------------------------------------------
    # selected_longitudes = np.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/selected_longitudes_pft_best.npy")
    # selected_latitudes  = np.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/selected_latitudes_pft_best.npy")
    # selected_coords     = np.column_stack((selected_longitudes, selected_latitudes))

    # -------------------------------------------------------------------------
    # 3) Prepare ERA5 file paths for each year
    # -------------------------------------------------------------------------
    root = Path("/Net/Groups/data_BGC/era5/e1/0d25_daily")
    ssrd_paths = sorted(
        (root / f"ssrd/ssrd.daily.fc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )
    vpd_paths = sorted(
        (root / f"vpd_cf/vpd_cf.daily.calc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )

    """
    subsample the data to increase the training speed
    """
    ssrd_path = ssrd_paths[0]
    ds_temp = xr.open_dataset(ssrd_path)
    # # land mask
    # land_mask = create_land_mask(ds_temp)
    # ds_temp = ds_temp.where(land_mask)

    # #eddy covariance mask
    # EC_mask = subsampling(selected_coords,ds_temp)
    # ds_temp = ds_temp.where(~EC_mask)

    # valid_mask = ~np.isnan(ds_temp['ssrd'].isel(time=0))
    # valid_pixels = np.argwhere(valid_mask.values)
    # num_selected = min(sample_size, len(valid_pixels))
    # selected_indices = np.random.choice(valid_pixels.shape[0], num_selected, replace=False)
    # selected_positions = valid_pixels[selected_indices,:] 
    # #Save longitude and latitude of selected pixels.
    # # Assuming overall_mean has 1D latitude and longitude coordinates with dimensions "latitude" then "longitude".
    # selected_lats = ds_temp.latitude.values[selected_positions[:,0]]
    # selected_lons = ds_temp.longitude.values[selected_positions[:,1]]
    # sample_coords = np.column_stack((selected_lats, selected_lons))
    # mask_da = subsampling(sample_coords,ds_temp)
    mask_da = xr.open_dataarray("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/subsample_mask_pre_training_vpd3_pft.nc")
    # mask_da.to_netcdf("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/subsample_mask_forward_"+case+".nc")
    # -------------------------------------------------------------------------
    # 4) Initialize the neural net and load weights
    # -------------------------------------------------------------------------
    if pft_code:
        evi_predictors = np.array([f"evi_{i}" for i in range(1, 5)])
        dynamic_predictors = ['ssrd']
        predictors = np.concatenate((dynamic_predictors, evi_predictors)) 
    else:
        predictors = ['ssrd']
    # input_dim = len(predictors)  # Number of predictors
    dynamic_input_dim = len(dynamic_predictors)
    static_input_dim = len(evi_predictors)
    model = NeuralNet(dynamic_input_dim+static_input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if pft_code:
        model.load_state_dict(torch.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/model_weights_pre_training_"+case+".pth"))
    else:
        model.load_state_dict(torch.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/model_weights_pre_training_"+case+".pth"))
    model.eval()

    target_gpp = 'sif'

    # -------------------------------------------------------------------------
    # 5) Prepare large aggregator arrays for pixel-wise statistics
    # -------------------------------------------------------------------------
    # We won't know the *exact* shape until we open at least one file and check
    # the coordinate dimension lengths. For safety:
    lat_vals = ds_temp.latitude.values  # shape ~ (720,)
    lon_vals = ds_temp.longitude.values # shape ~ (1440,)
    
    # ssrd_path = ssrd_paths[0]
    # vpd_path = vpd_paths[0]
    years_range = np.arange(start_year, end_year + 1)
    for year_idx, ssrd_path, vpd_path in zip(years_range, ssrd_paths, vpd_paths):
        # ---- Load & rename
        ds_ssrd = xr.open_dataset(ssrd_path) * 11.574
        ds_vpd = xr.open_dataset(vpd_path).rename({"vpd_cf": "vpd"})
        ds = xr.merge([ds_ssrd, ds_vpd])

        # ---- Compute GPP and SIF, plus uncertainties
        ds = SIF_GPP_EVI_model_ssrd(ds,use_f_ssrd)
        ds = SIF_GPP_EVI_model_ssrd_uncertainty(ds)

        ds['evi']  = (ds['evi']  - EVI_mean)   / EVI_std

        ds = compute_seasonal_pft_evi(ds)

        # ---- Standardize the input and the target
        ds["ssrd"] = (ds["ssrd"] - SW_IN_mean) / SW_IN_std
        # ds["vpd"]  = (ds["vpd"]  - VPD_mean)   / VPD_std
        ds['sif']  = (ds['sif']  - SIF_mean)   / SIF_std

        #subsample the data
        ds = ds.where(mask_da)

        # if pft_code:
        #     ds['is_tropical'] = xr.where(ds['pft'] == PFT['tropical'], 1.0, 0.0)
        #     ds['is_temperate'] = xr.where(ds['pft'] == PFT['temperate'], 1.0, 0.0)
        #     ds['is_boreal'] = xr.where(ds['pft'] == PFT['boreal'], 1.0, 0.0)

        # ---- Flatten data for PyTorch
        # We'll retrieve not just X_data, y_data but also the lat/lon indices.
        X_data, y_data, lat_idx_arr, lon_idx_arr, time_idx_arr = preprocess_data_lat_lon(
            ds, predictors, target_gpp
        )

        batch_size = 2**8
        dataset_climate = ClimateDataset_lat_lon(X_data, y_data, lat_idx_arr, lon_idx_arr, time_idx_arr)
        dataloader = DataLoader(dataset_climate, batch_size=batch_size, shuffle=False)

        pred_gpp_array = np.full((ds.dims["time"], ds.dims["latitude"], ds.dims["longitude"]), np.nan)

        with torch.no_grad():
            for batch_X, batch_y, lat_batch, lon_batch, time_batch in dataloader:
                outputs = model(batch_X.float().to(device))
                # Bring predictions and targets back to CPU
                outputs_np = outputs.cpu().numpy().squeeze()
                batch_y_np = batch_y.numpy()
                # Invert the standardization if you want metrics in original GPP units
                outputs_np_orig = outputs_np * SIF_std + SIF_mean
                batch_y_np_orig = batch_y_np * SIF_std + SIF_mean

                # Valid (non-NaN) mask
                valid_mask = ~np.isnan(outputs_np_orig) & ~np.isnan(batch_y_np_orig)
                outputs_valid = outputs_np_orig[valid_mask]
                lat_idx_valid = lat_batch[valid_mask]
                lon_idx_valid = lon_batch[valid_mask]
                time_idx_valid = time_batch[valid_mask]

                # Ensure lat_idx_valid and lon_idx_valid are numpy arrays
                if torch.is_tensor(lat_idx_valid):
                    lat_idx_valid = lat_idx_valid.cpu().numpy()
                if torch.is_tensor(lon_idx_valid):
                    lon_idx_valid = lon_idx_valid.cpu().numpy()

                lat_indices = np.array([np.where(lat_vals == val)[0][0] for val in lat_idx_valid])
                lon_indices = np.array([np.where(lon_vals == val)[0][0] for val in lon_idx_valid])

                # Assign predictions to the correct positions in pred_gpp_array
                pred_gpp_array[time_idx_valid, lat_indices, lon_indices] = outputs_valid

        ds['sif_hat'] = (("time","latitude","longitude"),pred_gpp_array)
        if pft_code:
            ds.to_netcdf("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/pre_training_forward_output_"+str(year_idx)+"_"+case+"_subsample.nc")
        else:
            ds.to_netcdf("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/pre_training_forward_output_"+str(year_idx)+"_"+case+"_subsample.nc")
        # Clear cache
        torch.cuda.empty_cache()
        print(f"Processed year file: {ssrd_path.name}")
        print_memory_usage()

def subsampling(sample_coords,ds):
    
    # get the range of the latitude and longitude
    lats = ds.latitude.values
    lons = ds.longitude.values

    # Create a meshgrid of coordinates (shape: [nlat, nlon])
    lat2d, lon2d = np.meshgrid(lats, lons)

    mask = np.zeros_like(lat2d, dtype=bool)
    for lat, lon in sample_coords:
        mask |= (lat2d == lat) & (lon2d == lon)

    # convert mask to a xarray DataArray
    mask_da = xr.DataArray(mask, coords=[("longitude", lons),("latitude", lats)])

    return mask_da


# function to pre training a neural network model
def pre_training_subsample(start_year=2017, end_year=2017,case='vpd3',
                           sample_size=10000,use_f_ssrd=False, pft_code=True):
    
    #read the global mean and std for standardization
    ds_stats = xr.open_dataset("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/combined_stats_land_"+case+".nc")

    root = Path("/Net/Groups/data_BGC/era5/e1/0d25_daily")
    ssrd_paths = sorted(
        (root / f"ssrd/ssrd.daily.fc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )
    vpd_paths = sorted(
        (root / f"vpd_cf/vpd_cf.daily.calc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )

    """
    subsample the data to increase the training speed
    """
    # ssrd_path = ssrd_paths[0]
    # ds_temp = xr.open_dataset(ssrd_path)
    # # land mask
    # land_mask = create_land_mask(ds_temp)
    # ds_temp = ds_temp.where(land_mask)

    # valid_mask = ~np.isnan(ds_temp['ssrd'].isel(time=0))
    # valid_pixels = np.argwhere(valid_mask.values)
    # num_selected = min(sample_size, len(valid_pixels))
    # selected_indices = np.random.choice(valid_pixels.shape[0], num_selected, replace=False)
    # selected_positions = valid_pixels[selected_indices,:] 
    # #Save longitude and latitude of selected pixels.
    # # Assuming overall_mean has 1D latitude and longitude coordinates with dimensions "latitude" then "longitude".
    # selected_lats = ds_temp.latitude.values[selected_positions[:,0]]
    # selected_lons = ds_temp.longitude.values[selected_positions[:,1]]
    # sample_coords = np.column_stack((selected_lats, selected_lons))
    # mask_da = subsampling(sample_coords,ds_temp)
    mask_da = xr.open_dataarray("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/subsample_mask_pre_training_vpd3_pft.nc")

    # Initialize model, loss function, and optimizer
    if pft_code:
        predictors = ['ssrd', 'is_tropical', 'is_temperate', 'is_boreal']    
    else:
        predictors = ['ssrd']
    input_dim = len(predictors)  # Number of predictors
    model = NeuralNet(input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model incrementally
    num_epochs = 10
    # ssrd_path = ssrd_paths[0]
    # vpd_path = vpd_paths[0]
    for ssrd_path, vpd_path in zip(ssrd_paths, vpd_paths):
        # Load one file at a time
        ds_ssrd = xr.open_dataset(ssrd_path)
        ds_ssrd = ds_ssrd * 11.574
        ds_vpd = xr.open_dataset(vpd_path)
        ds_vpd = ds_vpd.rename({"vpd_cf": "vpd"})
        ds = xr.merge([ds_ssrd, ds_vpd])

        # Calculate GPP and SIF
        ds = SIF_GPP_model_ssrd(ds,use_f_ssrd)
        # add uncertainty to the SIF and GPP
        ds = SIF_GPP_model_ssrd_uncertainty(ds)

        #subsampling
        ds = ds.where(mask_da)
        # add pft code
        # ds['pft_code'] = xr.where(ds['pft'] == 'tropical', 1,
        #                           xr.where(ds['pft'] == 'temperate', 2,
        #                           xr.where(ds['pft'] == 'boreal', 3, np.nan)))
        # Convert pft to one-hot encoded variables
        if pft_code:
            ds['is_tropical'] = xr.where(ds['pft'] == PFT['tropical'], 1.0, 0.0)
            ds['is_temperate'] = xr.where(ds['pft'] == PFT['temperate'], 1.0, 0.0)
            ds['is_boreal'] = xr.where(ds['pft'] == PFT['boreal'], 1.0, 0.0)
        
        # Standardize ssrd, vpd, and sif using the pre-calculated global mean and std
        ds["ssrd"] = (ds["ssrd"] - ds_stats['SW_IN_mean'].values) / ds_stats['SW_IN_std'].values
        # ds["vpd"] = (ds["vpd"] - ds_stats['VPD_mean'].values) / ds_stats['VPD_std'].values
        ds["sif"] = (ds["sif"] - ds_stats['SIF_mean'].values) / ds_stats['SIF_std'].values

        # Preprocess data
        target_sif = 'sif'
        X_train_full, y_train_full = preprocess_data(ds, predictors, target_sif)
        
        # Split data into training and validation sets
        # X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train_full, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_full, dtype=torch.float32).unsqueeze(1)  # Ensure correct shape
        # X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        # y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

        # Create DataLoaders
        batch_size = 2**8  # You can adjust this value based on your available memory and GPU capacity
        train_loader = DataLoader(ClimateDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
        #val_loader = DataLoader(ClimateDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)
        start_time = time.time()  # Start time

        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            # # Validation Step
            # model.eval()
            # total_val_loss = 0
            # with torch.no_grad():
            #     for batch_X, batch_y in val_loader:
            #         batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            #         outputs = model(batch_X)
            #         loss = criterion(outputs, batch_y)
            #         total_val_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            # avg_val_loss = total_val_loss / len(val_loader)

            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

            # Clear cache to free up memory
            torch.cuda.empty_cache()
        end_time = time.time()  # End time
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.2f} seconds")
        print_memory_usage()

    # Save the final model
    if pft_code:
        torch.save(model.state_dict(), "/Net/Groups/BGI/people/xyu/sifluxcom/outputs/model_weights_pre_training_"+case+"_one-hot_encoding.pth")
    else:
        torch.save(model.state_dict(), "/Net/Groups/BGI/people/xyu/sifluxcom/outputs/model_weights_pre_training_"+case+".pth")
    print('pre training finished!')

# function to pre training a neural network model
def pre_training_subsample_EVI(start_year=2017, end_year=2017,case='vpd3',
                           sample_size=10000,use_f_ssrd=False, pft_code=True):
    
    #read the global mean and std for standardization
    ds_stats = xr.open_dataset("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/combined_stats_land_"+case+".nc")

    root = Path("/Net/Groups/data_BGC/era5/e1/0d25_daily")
    ssrd_paths = sorted(
        (root / f"ssrd/ssrd.daily.fc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )
    vpd_paths = sorted(
        (root / f"vpd_cf/vpd_cf.daily.calc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )

    """
    subsample the data to increase the training speed
    """
    # ssrd_path = ssrd_paths[0]
    # ds_temp = xr.open_dataset(ssrd_path)
    # # land mask
    # land_mask = create_land_mask(ds_temp)
    # ds_temp = ds_temp.where(land_mask)

    # valid_mask = ~np.isnan(ds_temp['ssrd'].isel(time=0))
    # valid_pixels = np.argwhere(valid_mask.values)
    # num_selected = min(sample_size, len(valid_pixels))
    # selected_indices = np.random.choice(valid_pixels.shape[0], num_selected, replace=False)
    # selected_positions = valid_pixels[selected_indices,:] 
    # #Save longitude and latitude of selected pixels.
    # # Assuming overall_mean has 1D latitude and longitude coordinates with dimensions "latitude" then "longitude".
    # selected_lats = ds_temp.latitude.values[selected_positions[:,0]]
    # selected_lons = ds_temp.longitude.values[selected_positions[:,1]]
    # sample_coords = np.column_stack((selected_lats, selected_lons))
    # mask_da = subsampling(sample_coords,ds_temp)
    mask_da = xr.open_dataarray("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/subsample_mask_pre_training_vpd3_pft.nc")

    # Initialize model, loss function, and optimizer
    if pft_code:
        predictors = ['ssrd', 'evi']    
    else:
        predictors = ['ssrd']
    input_dim = len(predictors)  # Number of predictors
    model = NeuralNet(input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model incrementally
    num_epochs = 10
    # ssrd_path = ssrd_paths[0]
    # vpd_path = vpd_paths[0]
    for ssrd_path, vpd_path in zip(ssrd_paths, vpd_paths):
        # Load one file at a time
        ds_ssrd = xr.open_dataset(ssrd_path)
        ds_ssrd = ds_ssrd * 11.574
        ds_vpd = xr.open_dataset(vpd_path)
        ds_vpd = ds_vpd.rename({"vpd_cf": "vpd"})
        ds = xr.merge([ds_ssrd, ds_vpd])

        # Calculate GPP and SIF
        ds = SIF_GPP_EVI_model_ssrd(ds,use_f_ssrd)
        # add uncertainty to the SIF and GPP
        ds = SIF_GPP_EVI_model_ssrd_uncertainty(ds)

        #subsampling
        ds = ds.where(mask_da)
        # add pft code
        # ds['pft_code'] = xr.where(ds['pft'] == 'tropical', 1,
        #                           xr.where(ds['pft'] == 'temperate', 2,
        #                           xr.where(ds['pft'] == 'boreal', 3, np.nan)))
        # Convert pft to one-hot encoded variables
        # if pft_code:
        #     ds['is_tropical'] = xr.where(ds['pft'] == PFT['tropical'], 1.0, 0.0)
        #     ds['is_temperate'] = xr.where(ds['pft'] == PFT['temperate'], 1.0, 0.0)
        #     ds['is_boreal'] = xr.where(ds['pft'] == PFT['boreal'], 1.0, 0.0)
        
        # Standardize ssrd, vpd, and sif using the pre-calculated global mean and std
        ds["ssrd"] = (ds["ssrd"] - ds_stats['SW_IN_mean'].values) / ds_stats['SW_IN_std'].values
        # ds["vpd"] = (ds["vpd"] - ds_stats['VPD_mean'].values) / ds_stats['VPD_std'].values
        ds["sif"] = (ds["sif"] - ds_stats['SIF_mean'].values) / ds_stats['SIF_std'].values
        ds["evi"] = (ds["evi"] - ds_stats['EVI_mean'].values) / ds_stats['EVI_std'].values

        # Preprocess data
        target_sif = 'sif'
        X_train_full, y_train_full = preprocess_data(ds, predictors, target_sif)
        
        # Split data into training and validation sets
        # X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train_full, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_full, dtype=torch.float32).unsqueeze(1)  # Ensure correct shape
        # X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        # y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

        # Create DataLoaders
        batch_size = 2**8  # You can adjust this value based on your available memory and GPU capacity
        train_loader = DataLoader(ClimateDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
        #val_loader = DataLoader(ClimateDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)
        start_time = time.time()  # Start time

        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            # # Validation Step
            # model.eval()
            # total_val_loss = 0
            # with torch.no_grad():
            #     for batch_X, batch_y in val_loader:
            #         batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            #         outputs = model(batch_X)
            #         loss = criterion(outputs, batch_y)
            #         total_val_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            # avg_val_loss = total_val_loss / len(val_loader)

            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

            # Clear cache to free up memory
            torch.cuda.empty_cache()
        end_time = time.time()  # End time
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.2f} seconds")
        print_memory_usage()

    # Save the final model
    if pft_code:
        torch.save(model.state_dict(), "/Net/Groups/BGI/people/xyu/sifluxcom/outputs/model_weights_pre_training_"+case+".pth")
    else:
        torch.save(model.state_dict(), "/Net/Groups/BGI/people/xyu/sifluxcom/outputs/model_weights_pre_training_"+case+".pth")
    print('pre training finished!')

# function to pre training a neural network model
def pre_training_subsample_EVI_pft_mean(start_year=2017, end_year=2017,case='vpd3',
                           sample_size=10000,use_f_ssrd=False, pft_code=True):
    
    #read the global mean and std for standardization
    ds_stats = xr.open_dataset("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/combined_stats_land_pft_EVI.nc")

    root = Path("/Net/Groups/data_BGC/era5/e1/0d25_daily")
    ssrd_paths = sorted(
        (root / f"ssrd/ssrd.daily.fc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )
    vpd_paths = sorted(
        (root / f"vpd_cf/vpd_cf.daily.calc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )

    """
    subsample the data to increase the training speed
    """
    # ssrd_path = ssrd_paths[0]
    # ds_temp = xr.open_dataset(ssrd_path)
    # # land mask
    # land_mask = create_land_mask(ds_temp)
    # ds_temp = ds_temp.where(land_mask)

    # valid_mask = ~np.isnan(ds_temp['ssrd'].isel(time=0))
    # valid_pixels = np.argwhere(valid_mask.values)
    # num_selected = min(sample_size, len(valid_pixels))
    # selected_indices = np.random.choice(valid_pixels.shape[0], num_selected, replace=False)
    # selected_positions = valid_pixels[selected_indices,:] 
    # #Save longitude and latitude of selected pixels.
    # # Assuming overall_mean has 1D latitude and longitude coordinates with dimensions "latitude" then "longitude".
    # selected_lats = ds_temp.latitude.values[selected_positions[:,0]]
    # selected_lons = ds_temp.longitude.values[selected_positions[:,1]]
    # sample_coords = np.column_stack((selected_lats, selected_lons))
    # mask_da = subsampling(sample_coords,ds_temp)
    mask_da = xr.open_dataarray("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/subsample_mask_pre_training_vpd3_pft.nc")

    # Initialize model, loss function, and optimizer
    if pft_code:
        predictors = ['ssrd', 'evi']    
    else:
        predictors = ['ssrd']
    input_dim = len(predictors)  # Number of predictors
    model = NeuralNet(input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model incrementally
    num_epochs = 10
    # ssrd_path = ssrd_paths[0]
    # vpd_path = vpd_paths[0]
    for ssrd_path, vpd_path in zip(ssrd_paths, vpd_paths):
        # Load one file at a time
        ds_ssrd = xr.open_dataset(ssrd_path)
        ds_ssrd = ds_ssrd * 11.574
        ds_vpd = xr.open_dataset(vpd_path)
        ds_vpd = ds_vpd.rename({"vpd_cf": "vpd"})
        ds = xr.merge([ds_ssrd, ds_vpd])

        # Calculate GPP and SIF
        ds = SIF_GPP_EVI_model_ssrd(ds,use_f_ssrd)
        # add uncertainty to the SIF and GPP
        ds = SIF_GPP_EVI_model_ssrd_uncertainty(ds)

        # Compute the tropical EVI time series (average over longitude and latitude)
        evi_tropical_series = ds['evi'].where(ds['pft'] == PFT['tropical']).mean(dim=['longitude', 'latitude'])
        evi_tropical = evi_tropical_series.broadcast_like(ds['evi']).where(ds['pft'] == PFT['tropical'])
        # Compute the temperate EVI time series similarly
        evi_temperate_series = ds['evi'].where(ds['pft'] == PFT['temperate']).mean(dim=['longitude', 'latitude'])
        evi_temperate = evi_temperate_series.broadcast_like(ds['evi']).where(ds['pft'] == PFT['temperate'])
        # Compute the boreal EVI time series similarly
        evi_boreal_series = ds['evi'].where(ds['pft'] == PFT['boreal']).mean(dim=['longitude', 'latitude'])
        evi_boreal = evi_boreal_series.broadcast_like(ds['evi']).where(ds['pft'] == PFT['boreal'])
        # Combine the three EVI arrays along a new 'pft' dimension
        evi_merged = evi_tropical.combine_first(evi_temperate).combine_first(evi_boreal)
        
        ds['evi'] = evi_merged
        #subsampling
        ds = ds.where(mask_da)
        # add pft code
        # ds['pft_code'] = xr.where(ds['pft'] == 'tropical', 1,
        #                           xr.where(ds['pft'] == 'temperate', 2,
        #                           xr.where(ds['pft'] == 'boreal', 3, np.nan)))
        # Convert pft to one-hot encoded variables
        # if pft_code:
        #     ds['is_tropical'] = xr.where(ds['pft'] == PFT['tropical'], 1.0, 0.0)
        #     ds['is_temperate'] = xr.where(ds['pft'] == PFT['temperate'], 1.0, 0.0)
        #     ds['is_boreal'] = xr.where(ds['pft'] == PFT['boreal'], 1.0, 0.0)
        
        # Standardize ssrd, vpd, and sif using the pre-calculated global mean and std
        ds["ssrd"] = (ds["ssrd"] - ds_stats['SW_IN_mean'].values) / ds_stats['SW_IN_std'].values
        # ds["vpd"] = (ds["vpd"] - ds_stats['VPD_mean'].values) / ds_stats['VPD_std'].values
        ds["sif"] = (ds["sif"] - ds_stats['SIF_mean'].values) / ds_stats['SIF_std'].values
        ds["evi"] = (ds["evi"] - ds_stats['EVI_mean'].values) / ds_stats['EVI_std'].values

        # Preprocess data
        target_sif = 'sif'
        X_train_full, y_train_full = preprocess_data(ds, predictors, target_sif)
        
        # Split data into training and validation sets
        # X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train_full, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_full, dtype=torch.float32).unsqueeze(1)  # Ensure correct shape
        # X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        # y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

        # Create DataLoaders
        batch_size = 2**8  # You can adjust this value based on your available memory and GPU capacity
        train_loader = DataLoader(ClimateDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
        #val_loader = DataLoader(ClimateDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)
        start_time = time.time()  # Start time

        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            # # Validation Step
            # model.eval()
            # total_val_loss = 0
            # with torch.no_grad():
            #     for batch_X, batch_y in val_loader:
            #         batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            #         outputs = model(batch_X)
            #         loss = criterion(outputs, batch_y)
            #         total_val_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            # avg_val_loss = total_val_loss / len(val_loader)

            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

            # Clear cache to free up memory
            torch.cuda.empty_cache()
        end_time = time.time()  # End time
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.2f} seconds")
        print_memory_usage()

    # Save the final model
    if pft_code:
        torch.save(model.state_dict(), "/Net/Groups/BGI/people/xyu/sifluxcom/outputs/model_weights_pre_training_"+case+".pth")
    else:
        torch.save(model.state_dict(), "/Net/Groups/BGI/people/xyu/sifluxcom/outputs/model_weights_pre_training_"+case+".pth")
    print('pre training finished!')

def pre_training_subsample_EVI_encode(start_year=2017, end_year=2017,case='vpd3',
                           sample_size=10000,use_f_ssrd=False, pft_code=True):
    
    
    #read the global mean and std for standardization
    ds_stats = xr.open_dataset("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/combined_stats_land_pft_EVI.nc")

    root = Path("/Net/Groups/data_BGC/era5/e1/0d25_daily")
    ssrd_paths = sorted(
        (root / f"ssrd/ssrd.daily.fc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )
    vpd_paths = sorted(
        (root / f"vpd_cf/vpd_cf.daily.calc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )

    """
    subsample the data to increase the training speed
    """
    # ssrd_path = ssrd_paths[0]
    # ds_temp = xr.open_dataset(ssrd_path)
    # # land mask
    # land_mask = create_land_mask(ds_temp)
    # ds_temp = ds_temp.where(land_mask)

    # valid_mask = ~np.isnan(ds_temp['ssrd'].isel(time=0))
    # valid_pixels = np.argwhere(valid_mask.values)
    # num_selected = min(sample_size, len(valid_pixels))
    # selected_indices = np.random.choice(valid_pixels.shape[0], num_selected, replace=False)
    # selected_positions = valid_pixels[selected_indices,:] 
    # #Save longitude and latitude of selected pixels.
    # # Assuming overall_mean has 1D latitude and longitude coordinates with dimensions "latitude" then "longitude".
    # selected_lats = ds_temp.latitude.values[selected_positions[:,0]]
    # selected_lons = ds_temp.longitude.values[selected_positions[:,1]]
    # sample_coords = np.column_stack((selected_lats, selected_lons))
    # mask_da = subsampling(sample_coords,ds_temp)
    mask_da = xr.open_dataarray("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/subsample_mask_pre_training_vpd3_pft.nc")

    # Initialize model, loss function, and optimizer
    if pft_code:
        evi_predictors = np.array([f"evi_{i}" for i in range(1, 5)])
        dynamic_predictors = ['ssrd']
        predictors = np.concatenate((dynamic_predictors, evi_predictors)) 
    else:
        predictors = ['ssrd']
    # input_dim = len(predictors)  # Number of predictors
    dynamic_input_dim = len(dynamic_predictors)
    static_input_dim = len(evi_predictors)
    model = NeuralNet(dynamic_input_dim+static_input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model incrementally
    num_epochs = 10
    # ssrd_path = ssrd_paths[0]
    # vpd_path = vpd_paths[0]
    for ssrd_path, vpd_path in zip(ssrd_paths, vpd_paths):
        # Load one file at a time
        ds_ssrd = xr.open_dataset(ssrd_path)
        ds_ssrd = ds_ssrd * 11.574
        ds_vpd = xr.open_dataset(vpd_path)
        ds_vpd = ds_vpd.rename({"vpd_cf": "vpd"})
        ds = xr.merge([ds_ssrd, ds_vpd])

        # Calculate GPP and SIF
        ds = SIF_GPP_EVI_model_ssrd(ds,use_f_ssrd)
        # add uncertainty to the SIF and GPP
        ds = SIF_GPP_EVI_model_ssrd_uncertainty(ds)

        ds["evi"] = (ds["evi"] - ds_stats['EVI_mean'].values) / ds_stats['EVI_std'].values

        ds = compute_seasonal_pft_evi(ds)
        
        ds = ds.where(mask_da)

        #subsampling
        # add pft code
        # ds['pft_code'] = xr.where(ds['pft'] == 'tropical', 1,
        #                           xr.where(ds['pft'] == 'temperate', 2,
        #                           xr.where(ds['pft'] == 'boreal', 3, np.nan)))
        # Convert pft to one-hot encoded variables
        # if pft_code:
        #     ds['is_tropical'] = xr.where(ds['pft'] == PFT['tropical'], 1.0, 0.0)
        #     ds['is_temperate'] = xr.where(ds['pft'] == PFT['temperate'], 1.0, 0.0)
        #     ds['is_boreal'] = xr.where(ds['pft'] == PFT['boreal'], 1.0, 0.0)
        
        # Standardize ssrd, vpd, and sif using the pre-calculated global mean and std
        ds["ssrd"] = (ds["ssrd"] - ds_stats['SW_IN_mean'].values) / ds_stats['SW_IN_std'].values
        # ds["vpd"] = (ds["vpd"] - ds_stats['VPD_mean'].values) / ds_stats['VPD_std'].values
        ds["sif"] = (ds["sif"] - ds_stats['SIF_mean'].values) / ds_stats['SIF_std'].values

        # Preprocess data
        target_sif = 'sif'
        X_train_full, y_train_full = preprocess_data(ds, predictors, target_sif)
        
        # Split data into training and validation sets
        # X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

        # Convert data to PyTorch tensors
        X_tensor = torch.tensor(X_train_full, dtype=torch.float32)
        y_tensor = torch.tensor(y_train_full, dtype=torch.float32).unsqueeze(1)  # Ensure correct shape
        # X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        # y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

        # Create DataLoaders
        batch_size = 2**8  # You can adjust this value based on your available memory and GPU capacity
        train_loader = DataLoader(ClimateDataset(X_tensor,y_tensor), batch_size=batch_size, shuffle=True)
        #val_loader = DataLoader(ClimateDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)
        start_time = time.time()  # Start time

        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            # # Validation Step
            # model.eval()
            # total_val_loss = 0
            # with torch.no_grad():
            #     for batch_X, batch_y in val_loader:
            #         batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            #         outputs = model(batch_X)
            #         loss = criterion(outputs, batch_y)
            #         total_val_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            # avg_val_loss = total_val_loss / len(val_loader)

            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

            # Clear cache to free up memory
            torch.cuda.empty_cache()
        end_time = time.time()  # End time
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.2f} seconds")
        print_memory_usage()

    # Save the final model
    if pft_code:
        torch.save(model.state_dict(), "/Net/Groups/BGI/people/xyu/sifluxcom/outputs/model_weights_pre_training_"+case+".pth")
    else:
        torch.save(model.state_dict(), "/Net/Groups/BGI/people/xyu/sifluxcom/outputs/model_weights_pre_training_"+case+".pth")
    print('pre training finished!')

# create the dataset of randomly selected 300 pixels
def extract_selected_pixels(start_year=2012,end_year=2021,case='vpd3',use_f_ssrd=False):
    # Load the global mean and std for standardization
    ds_stats = xr.open_dataset("./outputs/combined_stats_land_"+case+".nc")

    # Load the selected pixel coordinates
    selected_longitudes = np.load("./outputs/selected_longitudes_pft_best.npy")
    selected_latitudes = np.load("./outputs/selected_latitudes_pft_best.npy")
    selected_coordinates = np.column_stack((selected_longitudes, selected_latitudes))

    root = Path("/Net/Groups/data_BGC/era5/e1/0d25_daily")
    ssrd_paths = sorted(
        (root / f"ssrd/ssrd.daily.fc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )
    vpd_paths = sorted(
        (root / f"vpd_cf/vpd_cf.daily.calc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )

    # Restrict to the desired year range
    year_list = np.arange(start_year, end_year + 1)
    year_datasets = []

    for ssrd_path, vpd_path in zip(ssrd_paths, vpd_paths):
        # Load one file at a time
        ds_ssrd = xr.open_dataset(ssrd_path)
        ds_ssrd = ds_ssrd * 11.574
        ds_vpd = xr.open_dataset(vpd_path)
        ds_vpd = ds_vpd.rename({"vpd_cf": "vpd"})
        ds = xr.merge([ds_ssrd, ds_vpd])
        
        # Calculate GPP and SIF
        ds = SIF_GPP_model_ssrd(ds,use_f_ssrd)
        # uncertainty
        ds = SIF_GPP_model_ssrd_uncertainty(ds)

        # Standardize ssrd, vpd, and sif using the pre-calculated global mean and std
        ds["ssrd"] = (ds["ssrd"] - ds_stats['SW_IN_mean'].values) / ds_stats['SW_IN_std'].values
        ds["vpd"] = (ds["vpd"] - ds_stats['VPD_mean'].values) / ds_stats['VPD_std'].values
        ds['gpp'] = (ds['gpp'] - ds_stats['GPP_mean'].values) / ds_stats['GPP_std'].values

        # Loop over selected coordinates and extract each pixel via nearest neighbor
        pixel_list = []
        i, (lon, lat) = next(enumerate(selected_coordinates))
        for i, (lon, lat) in enumerate(selected_coordinates):
            ds_pixel = ds.sel(longitude=lon, latitude=lat, method="nearest")
            # Label the pixel using a new 'pixel' dimension; this remains constant across years.
            ds_pixel = ds_pixel.expand_dims(pixel=[i])
            pixel_list.append(ds_pixel)
        
        # Concatenate the 300 pixels into one dataset (dimensions: time and pixel)
        ds_year_pixels = xr.concat(pixel_list, dim="pixel")
        # Instead of adding a new "year" dimension, we leave the time dimension intact.
        year_datasets.append(ds_year_pixels)

    # Combine all years along the existing "time" dimension.
    ds_selected = xr.concat(year_datasets, dim="time")

    # Optionally, save the result
    ds_selected.to_netcdf("./outputs/selected_pixels_"+str(start_year)+"_"+str(end_year)+"_"+case+".nc")
    print("Combined dataset saved.")

# create the dataset of randomly selected 300 pixels
def extract_selected_pixels_EVI(start_year=2012,end_year=2021,case='vpd3',use_f_ssrd=False):
    # Load the global mean and std for standardization
    ds_stats = xr.open_dataset("./outputs/combined_stats_land_"+case+".nc")

    # Load the selected pixel coordinates
    selected_longitudes = np.load("./outputs/selected_longitudes_pft_best.npy")
    selected_latitudes = np.load("./outputs/selected_latitudes_pft_best.npy")
    selected_coordinates = np.column_stack((selected_longitudes, selected_latitudes))

    root = Path("/Net/Groups/data_BGC/era5/e1/0d25_daily")
    ssrd_paths = sorted(
        (root / f"ssrd/ssrd.daily.fc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )
    vpd_paths = sorted(
        (root / f"vpd_cf/vpd_cf.daily.calc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )

    # Restrict to the desired year range
    year_list = np.arange(start_year, end_year + 1)
    year_datasets = []

    for ssrd_path, vpd_path in zip(ssrd_paths, vpd_paths):
        # Load one file at a time
        ds_ssrd = xr.open_dataset(ssrd_path)
        ds_ssrd = ds_ssrd * 11.574
        ds_vpd = xr.open_dataset(vpd_path)
        ds_vpd = ds_vpd.rename({"vpd_cf": "vpd"})
        ds = xr.merge([ds_ssrd, ds_vpd])
        
        # Calculate GPP and SIF
        ds = SIF_GPP_EVI_model_ssrd(ds,use_f_ssrd)
        # uncertainty
        ds = SIF_GPP_EVI_model_ssrd_uncertainty(ds)

        # Standardize ssrd, vpd, and sif using the pre-calculated global mean and std
        ds["ssrd"] = (ds["ssrd"] - ds_stats['SW_IN_mean'].values) / ds_stats['SW_IN_std'].values
        ds["vpd"] = (ds["vpd"] - ds_stats['VPD_mean'].values) / ds_stats['VPD_std'].values
        ds['gpp'] = (ds['gpp'] - ds_stats['GPP_mean'].values) / ds_stats['GPP_std'].values
        ds['evi'] = (ds['evi'] - ds_stats['EVI_mean'].values) / ds_stats['EVI_std'].values

        # Loop over selected coordinates and extract each pixel via nearest neighbor
        pixel_list = []
        i, (lon, lat) = next(enumerate(selected_coordinates))
        for i, (lon, lat) in enumerate(selected_coordinates):
            ds_pixel = ds.sel(longitude=lon, latitude=lat, method="nearest")
            # Label the pixel using a new 'pixel' dimension; this remains constant across years.
            ds_pixel = ds_pixel.expand_dims(pixel=[i])
            pixel_list.append(ds_pixel)
        
        # Concatenate the 300 pixels into one dataset (dimensions: time and pixel)
        ds_year_pixels = xr.concat(pixel_list, dim="pixel")
        # Instead of adding a new "year" dimension, we leave the time dimension intact.
        year_datasets.append(ds_year_pixels)

    # Combine all years along the existing "time" dimension.
    ds_selected = xr.concat(year_datasets, dim="time")

    # Optionally, save the result
    ds_selected.to_netcdf("./outputs/selected_pixels_"+str(start_year)+"_"+str(end_year)+"_"+case+".nc")
    print("Combined dataset saved.")

# function to run default training using the classic FLUXCOM approach
def default_training(case='vpd3'):
    selected_pixels = xr.open_dataset("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/selected_pixels_2012_2021_"+case+".nc")

    selected_pixels = xr.open_dataset("./outputs/selected_pixels_2012_2021_"+case+".nc")
    
    # Preprocess data
    predictors = ['ssrd']
    target_gpp = 'gpp'
    X_train_full, y_train_full = preprocess_data(selected_pixels, predictors, target_gpp)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Ensure correct shape
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    # Create DataLoaders
    batch_size = 2**6  # You can adjust this value based on your available memory and GPU capacity
    train_loader = DataLoader(ClimateDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ClimateDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    input_dim = len(predictors)  # Number of predictors (ssrd and vpd)
    model = NeuralNet(input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 150
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Validation Step
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.10f}, "
                f"Val Loss: {avg_val_loss:.10f}")

        # Clear cache to free up memory
        torch.cuda.empty_cache()

    # Save the final model
    # Save the final model
    torch.save(model.state_dict(), "./outputs/model_weights_default_"+case+".pth")
    print('default training finished!')

# function to forward run the default trained model
def default_forward(start_year=2012, end_year=2021,case='vpd3'):
    # Load the global mean and std for standardization
    ds_stats = xr.open_dataset("./outputs/combined_stats_land_"+case+".nc")

    # Load the selected pixel coordinates
    selected_longitudes = np.load("./outputs/selected_longitudes_"+case+".npy")
    selected_latitudes = np.load("./outputs/selected_latitudes_"+case+".npy")
    selected_coords = np.column_stack((selected_longitudes, selected_latitudes))

    root = Path("/Net/Groups/data_BGC/era5/e1/0d25_daily")
    ssrd_paths = sorted(
        (root / f"ssrd/ssrd.daily.fc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )
    vpd_paths = sorted(
        (root / f"vpd_cf/vpd_cf.daily.calc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )

    # Initialize model and load pre-training weights
    predictors = ['ssrd']
    input_dim = len(predictors)  # predictors: ssrd and vpd
    model = NeuralNet(input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load("./outputs/model_weights_default_"+case+".pth"))
    model.eval()

    target_gpp = 'gpp'

    # Initialize global accumulators
    global_squared_error = 0.0
    global_count = 0
    global_sum_y = 0.0
    global_sum_y_squared = 0.0

    # Loop over each year's files
    for ssrd_path, vpd_path in zip(ssrd_paths, vpd_paths):
        # Load one file at a time and merge datasets
        ds_ssrd = xr.open_dataset(ssrd_path)
        ds_ssrd = ds_ssrd * 11.574
        ds_vpd = xr.open_dataset(vpd_path)
        ds_vpd = ds_vpd.rename({"vpd_cf": "vpd"})
        ds = xr.merge([ds_ssrd, ds_vpd])
        
        # Calculate GPP and SIF
        ds = SIF_GPP_model(ds)
        # uncertainty  
        ds = SIF_GPP_model_uncertainty(ds)

        # Standardize ssrd and vpd using global stats
        ds["ssrd"] = (ds["ssrd"] - ds_stats['SW_IN_mean'].values) / ds_stats['SW_IN_std'].values
        ds["vpd"] = (ds["vpd"] - ds_stats['VPD_mean'].values) / ds_stats['VPD_std'].values
        ds['gpp'] = (ds['gpp'] - ds_stats['GPP_mean'].values) / ds_stats['GPP_std'].values

        # Extract coordinate values
        lon = ds.longitude.values   # shape (1440,)
        lat = ds.latitude.values     # shape (720,)

        # Create 2D meshgrid for lon and lat (note the indexing='ij' to match dims order)
        lon2d, lat2d = np.meshgrid(lon, lat, indexing='ij')  # both shape (1440, 720)
        #print(ds.dims)
        # Initialize a boolean mask (True where you want to remove the data)
        mask = np.zeros(lon2d.shape, dtype=bool)

        # Loop over each selected coordinate pair and update the mask.
        # Use np.isclose if the coordinates are floats to avoid precision issues.
        for selected in selected_coords:
            sel_lon, sel_lat = selected
            mask |= (np.isclose(lon2d, sel_lon)) & (np.isclose(lat2d, sel_lat))

        # Convert the mask into an xarray DataArray, associating the correct coords and dims.
        mask_da = xr.DataArray(mask, dims=["longitude", "latitude"],
                            coords={"longitude": lon, "latitude": lat})

        # Remove selected pixels using where (drop=True)
        ds_remaining = ds.where(~mask_da, drop=True)

        # Preprocess and get coordinates
        X_data, y_data = preprocess_data(ds_remaining, predictors, target_gpp)
        
        dataset_climate = ClimateDataset(X_data, y_data)
        dataloader = DataLoader(dataset_climate, batch_size=64, shuffle=False)
        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                # Squeeze outputs to remove the extra dimension (shape becomes (batch_size,))
                outputs_np = outputs.cpu().numpy().squeeze()
                batch_y_np = batch_y.numpy()
                
                # Now both outputs_np and batch_y_np have the same shape
                valid = ~np.isnan(outputs_np) & ~np.isnan(batch_y_np)
                outputs_valid = outputs_np[valid]
                batch_y_valid = batch_y_np[valid]
                
                # Update global accumulators with valid values
                diff = outputs_valid - batch_y_valid
                global_squared_error += np.sum(diff**2)
                n = batch_y_valid.size
                global_count += n
                global_sum_y += np.sum(batch_y_valid)
                global_sum_y_squared += np.sum(batch_y_valid**2)
        
        # Clear cache to free up memory
        torch.cuda.empty_cache()
        print(f"Processed year file: {ssrd_path.name}")
        print_memory_usage()

    # Compute overall RMSE and R2 using global accumulators
    overall_rmse = np.sqrt(global_squared_error / global_count)
    # global_mean_y = global_sum_y / global_count
    sst = global_sum_y_squared - (global_sum_y**2) / global_count
    overall_r2 = 1 - global_squared_error / sst
    print("Overall RMSE:", overall_rmse)
    print("Overall R2:", overall_r2)

    print('Default forward run finished!')


def default_forward_pixelwise(start_year=2012, end_year=2013,case='vpd3',use_f_ssrd=False):
    # -------------------------------------------------------------------------
    # 1) Load global mean/std for standardization
    # -------------------------------------------------------------------------
    ds_stats = xr.open_dataset("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/combined_stats_land_"+case+".nc")
    GPP_mean = ds_stats['GPP_mean'].values.item()
    GPP_std  = ds_stats['GPP_std'].values.item()

    SW_IN_mean = ds_stats['SW_IN_mean'].values.item()
    SW_IN_std  = ds_stats['SW_IN_std'].values.item()
    VPD_mean   = ds_stats['VPD_mean'].values.item()
    VPD_std    = ds_stats['VPD_std'].values.item()

    # -------------------------------------------------------------------------
    # 2) Load the selected pixel coordinates to *exclude* from evaluation
    # -------------------------------------------------------------------------
    selected_longitudes = np.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/selected_longitudes_"+case+".npy")
    selected_latitudes  = np.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/selected_latitudes_"+case+".npy")
    selected_coords     = np.column_stack((selected_longitudes, selected_latitudes))

    # -------------------------------------------------------------------------
    # 3) Prepare ERA5 file paths for each year
    # -------------------------------------------------------------------------
    root = Path("/Net/Groups/data_BGC/era5/e1/0d25_daily")
    ssrd_paths = sorted(
        (root / f"ssrd/ssrd.daily.fc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )
    vpd_paths = sorted(
        (root / f"vpd_cf/vpd_cf.daily.calc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )

    # -------------------------------------------------------------------------
    # 4) Initialize the neural net and load weights
    # -------------------------------------------------------------------------
    input_dim = 2  # we have 2 predictors: ssrd and vpd
    model = NeuralNet(input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/model_weights_default_"+case+".pth"))
    model.eval()

    predictors = ['ssrd', 'vpd']
    target_gpp = 'gpp'

    # -------------------------------------------------------------------------
    # 5) Prepare large aggregator arrays for pixel-wise statistics
    #    We'll assume the domain is 720 lat × 1440 lon (as per your data).
    #    We will fill them incrementally over time.
    # -------------------------------------------------------------------------
    # We won't know the *exact* shape until we open at least one file and check
    # the coordinate dimension lengths. For safety:
    sample_ds = xr.open_dataset(ssrd_paths[0])
    lat_vals = sample_ds.latitude.values  # shape ~ (720,)
    lon_vals = sample_ds.longitude.values # shape ~ (1440,)
    n_lat = lat_vals.size
    n_lon = lon_vals.size
    
    # Each aggregator is 2D: [n_lat, n_lon]
    sum_y   = np.zeros((n_lat, n_lon), dtype=np.float64)
    sum_y2  = np.zeros((n_lat, n_lon), dtype=np.float64)
    sum_diff   = np.zeros((n_lat, n_lon), dtype=np.float64)
    count   = np.zeros((n_lat, n_lon), dtype=np.float64)

    # -------------------------------------------------------------------------
    # 6) Loop over each year, build dataset, run model predictions in batches
    # -------------------------------------------------------------------------
    # (We will still keep track of global accumulators to see overall RMSE, R2)
    global_squared_error = 0.0
    global_count = 0
    global_sum_y = 0.0
    global_sum_y_squared = 0.0

    # ssrd_path = ssrd_paths[0]
    # vpd_path = vpd_paths[0]
    year_idx = start_year
    for ssrd_path, vpd_path in zip(ssrd_paths, vpd_paths):
        # ---- Load & rename
        ds_ssrd = xr.open_dataset(ssrd_path) * 11.574
        ds_vpd = xr.open_dataset(vpd_path).rename({"vpd_cf": "vpd"})
        ds = xr.merge([ds_ssrd, ds_vpd])

        # ---- Compute GPP and SIF, plus uncertainties
        ds = SIF_GPP_model(ds,use_f_ssrd)
        ds = SIF_GPP_model_uncertainty(ds)

        # ---- Standardize the input and the target
        ds["ssrd"] = (ds["ssrd"] - SW_IN_mean) / SW_IN_std
        ds["vpd"]  = (ds["vpd"]  - VPD_mean)   / VPD_std
        ds['gpp']  = (ds['gpp']  - GPP_mean)   / GPP_std

        # ---- Exclude your selected coordinates
        lon_1d = ds.longitude.values  # shape (1440,)
        lat_1d = ds.latitude.values   # shape (720,)
        lon2d, lat2d = np.meshgrid(lon_1d, lat_1d, indexing='ij')  # (1440, 720)
        
        mask = np.zeros(lon2d.shape, dtype=bool)
        for sel_lon, sel_lat in selected_coords:
            mask |= (np.isclose(lon2d, sel_lon)) & (np.isclose(lat2d, sel_lat))
        # Convert to DataArray
        mask_da = xr.DataArray(mask, dims=["longitude", "latitude"],
                               coords={"longitude": lon_1d, "latitude": lat_1d})

        land_mask = create_land_mask(ds)
        ds_land = ds.where(land_mask)

        ds_remaining = ds_land.where(~mask_da, drop=True)

        # ---- Flatten data for PyTorch
        # We'll retrieve not just X_data, y_data but also the lat/lon indices.
        X_data, y_data, lat_idx_arr, lon_idx_arr, time_idx_arr = preprocess_data_lat_lon(
            ds_remaining, predictors, target_gpp
        )

        dataset_climate = ClimateDataset_lat_lon(X_data, y_data, lat_idx_arr, lon_idx_arr, time_idx_arr)
        dataloader = DataLoader(dataset_climate, batch_size=2**15, shuffle=False)

        pred_gpp_array = np.full((ds.dims["time"], ds.dims["latitude"], ds.dims["longitude"]), np.nan)

        with torch.no_grad():
            for batch_X, batch_y, lat_batch, lon_batch, time_batch in dataloader:
                outputs = model(batch_X.float().to(device))
                # Bring predictions and targets back to CPU
                outputs_np = outputs.cpu().numpy().squeeze()
                batch_y_np = batch_y.numpy()
                # Invert the standardization if you want metrics in original GPP units
                outputs_np_orig = outputs_np * GPP_std + GPP_mean
                batch_y_np_orig = batch_y_np * GPP_std + GPP_mean

                # Valid (non-NaN) mask
                valid_mask = ~np.isnan(outputs_np_orig) & ~np.isnan(batch_y_np_orig)
                outputs_valid = outputs_np_orig[valid_mask]
                batch_y_valid = batch_y_np_orig[valid_mask]
                lat_idx_valid = lat_batch[valid_mask]
                lon_idx_valid = lon_batch[valid_mask]
                time_idx_valid = time_batch[valid_mask]

                # ---- Update global accumulators
                diff = outputs_valid - batch_y_valid
                global_squared_error += np.sum(diff**2)
                n_valid = batch_y_valid.size
                global_count += n_valid
                global_sum_y += np.sum(batch_y_valid)
                global_sum_y_squared += np.sum(batch_y_valid**2)

                # ---- Update pixel-wise accumulators
                # Ensure lat_idx_valid and lon_idx_valid are numpy arrays
                if torch.is_tensor(lat_idx_valid):
                    lat_idx_valid = lat_idx_valid.cpu().numpy()
                if torch.is_tensor(lon_idx_valid):
                    lon_idx_valid = lon_idx_valid.cpu().numpy()

                lat_indices = np.argmin(np.abs(lat_vals[:, None] - lat_idx_valid[None, :]), axis=0)
                lon_indices = np.argmin(np.abs(lon_vals[:, None] - lon_idx_valid[None, :]), axis=0)

                # Assign predictions to the correct positions in pred_gpp_array
                pred_gpp_array[time_idx_valid, lat_indices, lon_indices] = outputs_valid

                # Update pixel-wise accumulators with vectorized operations
                np.add.at(sum_y,   (lat_indices, lon_indices), batch_y_valid)
                np.add.at(sum_y2,  (lat_indices, lon_indices), batch_y_valid**2)
                np.add.at(sum_diff,(lat_indices, lon_indices), (batch_y_valid - outputs_valid)**2)
                np.add.at(count,   (lat_indices, lon_indices), 1)


        ds_land['gpp_hat'] = (("time","latitude","longitude"),pred_gpp_array)
        ds_land.to_netcdf("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/default_forward_output_"+str(year_idx)+"_"+case+".nc")
        # Clear cache
        torch.cuda.empty_cache()
        print(f"Processed year file: {ssrd_path.name}")
        print_memory_usage()

    # -------------------------------------------------------------------------
    # 7) Final overall RMSE and R² (global) if you still want them
    # -------------------------------------------------------------------------
    overall_rmse = np.sqrt(global_squared_error / global_count)
    sst = global_sum_y_squared - (global_sum_y**2) / global_count
    overall_r2 = 1 - global_squared_error / sst if sst > 0 else np.nan

    print("Overall RMSE:", overall_rmse)
    print("Overall R2:", overall_r2)
    print('Finished all years!')

    # -------------------------------------------------------------------------
    # 8) Compute pixel-wise RMSE and R²
    # -------------------------------------------------------------------------
    # SSE = sum(y - f)^2
    SSE = sum_diff
    # SST = sum(y^2) - (sum(y)^2 / n)
    SST = sum_y2 - (sum_y**2)/np.maximum(count, 1e-12)

    # We'll avoid divide-by-zero by masking out places with count=0 or SST=0
    valid_pix = (count > 0.5) & (SST > 1e-12)

    pixel_rmse = np.full((n_lat, n_lon), np.nan, dtype=np.float64)
    pixel_r2   = np.full((n_lat, n_lon), np.nan, dtype=np.float64)

    pixel_rmse[valid_pix] = np.sqrt( SSE[valid_pix] / count[valid_pix] )
    pixel_r2[valid_pix]   = 1.0 - (SSE[valid_pix] / SST[valid_pix])

    # -------------------------------------------------------------------------
    # 9) Save pixel-wise RMSE, R2 to NetCDF
    # -------------------------------------------------------------------------
    # Build an xarray Dataset
    ds_out = xr.Dataset(
        {
            "RMSE": (("latitude", "longitude"), pixel_rmse),
            "R2":   (("latitude", "longitude"), pixel_r2)
        },
        coords = {
            "latitude":  (("latitude",), lat_vals),
            "longitude": (("longitude",), lon_vals)
        }
    )

    # Example: save to "pixel_metrics.nc"
    ds_out.to_netcdf("./outputs/pixel_metrics_default_"+case+".nc")
    print("Saved pixel-wise metrics to ./outputs/pixel_metrics_default_"+case+".nc")

def default_forward_subsample(start_year=2012, end_year=2012,case='vpd3',sample_size=1000,use_f_ssrd=False):
    # case = 'vpd3_pft'
    # use_f_ssrd = True
    # -------------------------------------------------------------------------
    # 1) Load global mean/std for standardization
    # -------------------------------------------------------------------------
    ds_stats = xr.open_dataset("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/combined_stats_land_"+case+".nc")
    GPP_mean = ds_stats['GPP_mean'].values.item()
    GPP_std  = ds_stats['GPP_std'].values.item()

    SW_IN_mean = ds_stats['SW_IN_mean'].values.item()
    SW_IN_std  = ds_stats['SW_IN_std'].values.item()
    VPD_mean   = ds_stats['VPD_mean'].values.item()
    VPD_std    = ds_stats['VPD_std'].values.item()

    # -------------------------------------------------------------------------
    # 2) Load the selected pixel coordinates to *exclude* from evaluation
    # -------------------------------------------------------------------------
    selected_longitudes = np.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/selected_longitudes_"+case+".npy")
    selected_latitudes  = np.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/selected_latitudes_"+case+".npy")
    selected_coords     = np.column_stack((selected_longitudes, selected_latitudes))

    # -------------------------------------------------------------------------
    # 3) Prepare ERA5 file paths for each year
    # -------------------------------------------------------------------------
    root = Path("/Net/Groups/data_BGC/era5/e1/0d25_daily")
    ssrd_paths = sorted(
        (root / f"ssrd/ssrd.daily.fc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )
    vpd_paths = sorted(
        (root / f"vpd_cf/vpd_cf.daily.calc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )

    """
    subsample the data to increase the training speed
    """
    if use_f_ssrd:
        ssrd_path = ssrd_paths[0]
        vpd_path = vpd_paths[0]
        ds_ssrd_temp = xr.open_dataset(ssrd_path)
        ds_ssrd_temp = ds_ssrd_temp * 11.574
        ds_vpd_temp = xr.open_dataset(vpd_path)
        ds_vpd_temp = ds_vpd_temp.rename({"vpd_cf": "vpd"})
        ds_temp = xr.merge([ds_ssrd_temp, ds_vpd_temp])
        
        # Calculate GPP and SIF
        ds_temp = SIF_GPP_model_ssrd(ds_temp,use_f_ssrd)
        # uncertainty
        ds_temp = SIF_GPP_model_ssrd_uncertainty(ds_temp)

        # # land mask
        land_mask = create_land_mask(ds_temp)
        ds_temp = ds_temp.where(land_mask)

        pft = ds_temp['pft'].isel(time=0)
        mask = (pft == PFT['tropical']) | (pft == PFT['boreal'])
        valid_pixels = np.argwhere(mask.values)
        tropical_boreal_lats = pft.coords['latitude'][valid_pixels[:, 0]].values
        tropical_boreal_lons = pft.coords['longitude'][valid_pixels[:, 1]].values
        
        num_selected = min(sample_size, valid_pixels.shape[0])
        selected_indices = np.random.choice(valid_pixels.shape[0], num_selected, replace=False)
        selected_positions = valid_pixels[selected_indices,:] 
        #Save longitude and latitude of selected pixels.
        # Assuming overall_mean has 1D latitude and longitude coordinates with dimensions "latitude" then "longitude".
        selected_lats = ds_temp.latitude.values[selected_positions[:,0]]
        selected_lons = ds_temp.longitude.values[selected_positions[:,1]]
        sample_coords = np.column_stack((selected_lats, selected_lons))
        mask_da = subsampling(sample_coords,ds_temp)

        mask_da.to_netcdf("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/subsample_mask_forward_"+case+".nc")

    else:
        ssrd_path = ssrd_paths[0]
        ds_temp = xr.open_dataset(ssrd_path)
        # # land mask
        land_mask = create_land_mask(ds_temp)
        ds_temp = ds_temp.where(land_mask)

        # #eddy covariance mask
        EC_mask = subsampling(selected_coords,ds_temp)
        ds_temp = ds_temp.where(~EC_mask)
        mask_da = xr.open_dataarray("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/subsample_mask_forward_"+case+".nc")

    # valid_mask = ~np.isnan(ds_temp['ssrd'].isel(time=0))
    # valid_pixels = np.argwhere(valid_mask.values)
    # num_selected = min(sample_size, len(valid_pixels))
    # selected_indices = np.random.choice(valid_pixels.shape[0], num_selected, replace=False)
    # selected_positions = valid_pixels[selected_indices,:] 
    # #Save longitude and latitude of selected pixels.
    # # Assuming overall_mean has 1D latitude and longitude coordinates with dimensions "latitude" then "longitude".
    # selected_lats = ds_temp.latitude.values[selected_positions[:,0]]
    # selected_lons = ds_temp.longitude.values[selected_positions[:,1]]
    # sample_coords = np.column_stack((selected_lats, selected_lons))
    # mask_da = subsampling(sample_coords,ds_temp)
    # mask_da.to_netcdf("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/subsample_mask_forward_"+case+".nc")
    # -------------------------------------------------------------------------
    # 4) Initialize the neural net and load weights
    # -------------------------------------------------------------------------
    predictors = ['ssrd']
    input_dim = len(predictors)  # we have 2 predictors: ssrd and vpd
    model = NeuralNet(input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/model_weights_default_"+case+".pth"))
    model.eval()

    target_gpp = 'gpp'

    # -------------------------------------------------------------------------
    # 5) Prepare large aggregator arrays for pixel-wise statistics
    # -------------------------------------------------------------------------
    # We won't know the *exact* shape until we open at least one file and check
    # the coordinate dimension lengths. For safety:
    lat_vals = ds_temp.latitude.values  # shape ~ (720,)
    lon_vals = ds_temp.longitude.values # shape ~ (1440,)
    
    # ssrd_path = ssrd_paths[0]
    # vpd_path = vpd_paths[0]
    years_range = np.arange(start_year, end_year + 1)
    for year_idx, ssrd_path, vpd_path in zip(years_range, ssrd_paths, vpd_paths):
        # ---- Load & rename
        ds_ssrd = xr.open_dataset(ssrd_path) * 11.574
        ds_vpd = xr.open_dataset(vpd_path).rename({"vpd_cf": "vpd"})
        ds = xr.merge([ds_ssrd, ds_vpd])

        # ---- Compute GPP and SIF, plus uncertainties
        ds = SIF_GPP_model_ssrd(ds,use_f_ssrd)
        ds = SIF_GPP_model_ssrd_uncertainty(ds)

        # ---- Standardize the input and the target
        ds["ssrd"] = (ds["ssrd"] - SW_IN_mean) / SW_IN_std
        ds["vpd"]  = (ds["vpd"]  - VPD_mean)   / VPD_std
        ds['gpp']  = (ds['gpp']  - GPP_mean)   / GPP_std

        #subsample the data
        ds_remaining = ds.where(mask_da)

        # ---- Flatten data for PyTorch
        # We'll retrieve not just X_data, y_data but also the lat/lon indices.
        X_data, y_data, lat_idx_arr, lon_idx_arr, time_idx_arr = preprocess_data_lat_lon(
            ds_remaining, predictors, target_gpp
        )

        batch_size = 2**8
        dataset_climate = ClimateDataset_lat_lon(X_data, y_data, lat_idx_arr, lon_idx_arr, time_idx_arr)
        dataloader = DataLoader(dataset_climate, batch_size=batch_size, shuffle=False)

        pred_gpp_array = np.full((ds.dims["time"], ds.dims["latitude"], ds.dims["longitude"]), np.nan)

        with torch.no_grad():
            for batch_X, batch_y, lat_batch, lon_batch, time_batch in dataloader:
                outputs = model(batch_X.to(device))
                # Bring predictions and targets back to CPU
                outputs_np = outputs.cpu().numpy().squeeze()
                batch_y_np = batch_y.numpy()
                # Invert the standardization if you want metrics in original GPP units
                outputs_np_orig = outputs_np * GPP_std + GPP_mean
                batch_y_np_orig = batch_y_np * GPP_std + GPP_mean

                # Valid (non-NaN) mask
                valid_mask = ~np.isnan(outputs_np_orig) & ~np.isnan(batch_y_np_orig)
                outputs_valid = outputs_np_orig[valid_mask]
                lat_idx_valid = lat_batch[valid_mask]
                lon_idx_valid = lon_batch[valid_mask]
                time_idx_valid = time_batch[valid_mask]

                # Ensure lat_idx_valid and lon_idx_valid are numpy arrays
                if torch.is_tensor(lat_idx_valid):
                    lat_idx_valid = lat_idx_valid.cpu().numpy()
                if torch.is_tensor(lon_idx_valid):
                    lon_idx_valid = lon_idx_valid.cpu().numpy()

                lat_indices = np.array([np.where(lat_vals == val)[0][0] for val in lat_idx_valid])
                lon_indices = np.array([np.where(lon_vals == val)[0][0] for val in lon_idx_valid])

                # Assign predictions to the correct positions in pred_gpp_array
                pred_gpp_array[time_idx_valid, lat_indices, lon_indices] = outputs_valid

        ds_remaining['gpp_hat'] = (("time","latitude","longitude"),pred_gpp_array)
        ds_remaining.to_netcdf("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/default_forward_output_"+str(year_idx)+"_"+case+"_subsample.nc")
        # Clear cache
        torch.cuda.empty_cache()
        print(f"Processed year file: {ssrd_path.name}")
        print_memory_usage()


def default_forward_pixelwise_gpu(start_year=2012, end_year=2013,case='vpd3',use_f_ssrd=False):
    # -------------------------------------------------------------------------
    # 1) Load global mean/std for standardization
    # -------------------------------------------------------------------------
    ds_stats = xr.open_dataset("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/combined_stats_land_"+case+".nc")
    GPP_mean = ds_stats['GPP_mean'].values.item()
    GPP_std  = ds_stats['GPP_std'].values.item()

    SW_IN_mean = ds_stats['SW_IN_mean'].values.item()
    SW_IN_std  = ds_stats['SW_IN_std'].values.item()
    VPD_mean   = ds_stats['VPD_mean'].values.item()
    VPD_std    = ds_stats['VPD_std'].values.item()

    # -------------------------------------------------------------------------
    # 2) Load the selected pixel coordinates to *exclude* from evaluation
    # -------------------------------------------------------------------------
    selected_longitudes = np.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/selected_longitudes_"+case+".npy")
    selected_latitudes  = np.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/selected_latitudes_"+case+".npy")
    selected_coords     = np.column_stack((selected_longitudes, selected_latitudes))

    # -------------------------------------------------------------------------
    # 3) Prepare ERA5 file paths for each year
    # -------------------------------------------------------------------------
    root = Path("/Net/Groups/data_BGC/era5/e1/0d25_daily")
    ssrd_paths = sorted(
        (root / f"ssrd/ssrd.daily.fc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )
    vpd_paths = sorted(
        (root / f"vpd_cf/vpd_cf.daily.calc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )

    # -------------------------------------------------------------------------
    # 4) Initialize the neural net and load weights
    # -------------------------------------------------------------------------
    input_dim = 2  # we have 2 predictors: ssrd and vpd
    model = NeuralNet(input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/model_weights_default_"+case+".pth"))
    model.eval()

    predictors = ['ssrd', 'vpd']
    target_gpp = 'gpp'

    # -------------------------------------------------------------------------
    # 5) Prepare large aggregator arrays for pixel-wise statistics
    #    We'll assume the domain is 720 lat × 1440 lon (as per your data).
    #    We will fill them incrementally over time.
    # -------------------------------------------------------------------------
    # We won't know the *exact* shape until we open at least one file and check
    # the coordinate dimension lengths. For safety:
    sample_ds = xr.open_dataset(ssrd_paths[0])
    lat_vals = sample_ds.latitude.values  # shape ~ (720,)
    lon_vals = sample_ds.longitude.values # shape ~ (1440,)
    n_lat = lat_vals.size
    n_lon = lon_vals.size
    
    # Each aggregator is 2D: [n_lat, n_lon]
    sum_y   = np.zeros((n_lat, n_lon), dtype=np.float64)
    sum_y2  = np.zeros((n_lat, n_lon), dtype=np.float64)
    sum_diff   = np.zeros((n_lat, n_lon), dtype=np.float64)
    count   = np.zeros((n_lat, n_lon), dtype=np.float64)

    # -------------------------------------------------------------------------
    # 6) Loop over each year, build dataset, run model predictions in batches
    # -------------------------------------------------------------------------
    # (We will still keep track of global accumulators to see overall RMSE, R2)
    global_squared_error = 0.0
    global_count = 0
    global_sum_y = 0.0
    global_sum_y_squared = 0.0

    ssrd_path = ssrd_paths[0]
    vpd_path = vpd_paths[0]
    for ssrd_path, vpd_path in zip(ssrd_paths, vpd_paths):
        # ---- Load & rename
        ds_ssrd = xr.open_dataset(ssrd_path) * 11.574
        ds_vpd = xr.open_dataset(vpd_path).rename({"vpd_cf": "vpd"})
        ds = xr.merge([ds_ssrd, ds_vpd])

        # ---- Compute GPP and SIF, plus uncertainties
        ds = SIF_GPP_model(ds,use_f_ssrd)
        ds = SIF_GPP_model_uncertainty(ds)

        # ---- Standardize the input and the target
        ds["ssrd"] = (ds["ssrd"] - SW_IN_mean) / SW_IN_std
        ds["vpd"]  = (ds["vpd"]  - VPD_mean)   / VPD_std
        ds['gpp']  = (ds['gpp']  - GPP_mean)   / GPP_std

        # ---- Exclude your selected coordinates
        lon_1d = ds.longitude.values  # shape (1440,)
        lat_1d = ds.latitude.values   # shape (720,)
        lon2d, lat2d = np.meshgrid(lon_1d, lat_1d, indexing='ij')  # (1440, 720)
        
        mask = np.zeros(lon2d.shape, dtype=bool)
        for sel_lon, sel_lat in selected_coords:
            mask |= (np.isclose(lon2d, sel_lon)) & (np.isclose(lat2d, sel_lat))
        # Convert to DataArray
        mask_da = xr.DataArray(mask, dims=["longitude", "latitude"],
                               coords={"longitude": lon_1d, "latitude": lat_1d})

        ds_remaining = ds.where(~mask_da, drop=True)

        # ---- Flatten data for PyTorch
        # We'll retrieve not just X_data, y_data but also the lat/lon indices.
        X_data, y_data, lat_idx_arr, lon_idx_arr = preprocess_data_lat_lon(
            ds_remaining, predictors, target_gpp
        )

        dataset_climate = ClimateDataset_lat_lon(X_data, y_data, lat_idx_arr, lon_idx_arr)
        dataloader = DataLoader(dataset_climate, batch_size=256, shuffle=False)

        with torch.no_grad():
            # Move mean and std to GPU as tensors (if they aren't already)
            GPP_mean_t = torch.tensor(GPP_mean, dtype=torch.float32, device=device)
            GPP_std_t  = torch.tensor(GPP_std,  dtype=torch.float32, device=device)
            lat_vals_t = torch.tensor(lat_vals, dtype=torch.float32, device=device)
            lon_vals_t = torch.tensor(lon_vals, dtype=torch.float32, device=device)
            for batch_X, batch_y, lat_batch, lon_batch in dataloader:
            # Send all data to GPU
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                lat_batch = lat_batch.to(device)
                lon_batch = lon_batch.to(device)

                outputs = model(batch_X)  # Already on GPU
                # Undo standardization on GPU
                outputs = outputs.squeeze() * GPP_std_t + GPP_mean_t
                batch_y = batch_y * GPP_std_t + GPP_mean_t

                # Find valid (non-NaN) entries on GPU
                valid_mask = ~torch.isnan(outputs) & ~torch.isnan(batch_y)
                outputs_valid = outputs[valid_mask]
                batch_y_valid = batch_y[valid_mask]
                lat_valid = lat_batch[valid_mask]
                lon_valid = lon_batch[valid_mask]

                # Update global accumulators in GPU, then convert to float for .item()
                diff = outputs_valid - batch_y_valid
                global_squared_error += diff.pow(2).sum().item()
                n_valid = diff.numel()
                global_count += n_valid
                global_sum_y += batch_y_valid.sum().item()
                global_sum_y_squared += batch_y_valid.pow(2).sum().item()

                # Convert lat/lon arrays to 1D differences, find nearest index via GPU
                # shape: (n_lat, n_valid)
                lat_dists = (lat_vals_t.unsqueeze(1) - lat_valid.unsqueeze(0)).abs()
                lon_dists = (lon_vals_t.unsqueeze(1) - lon_valid.unsqueeze(0)).abs()
                lat_indices = lat_dists.argmin(dim=0)
                lon_indices = lon_dists.argmin(dim=0)

                # Accumulate pixel-wise metrics on GPU, then move to CPU
                # Because we want to accumulate possibly multiple hits to the same pixel,
                # do it in a Python loop to avoid collisions:
                for i in range(n_valid):
                    li = lat_indices[i].item()
                    lj = lon_indices[i].item()
                    y_val = batch_y_valid[i].item()
                    diff_sq = (batch_y_valid[i] - outputs_valid[i]).pow(2).item()

                    sum_y[li, lj]    += y_val
                    sum_y2[li, lj]   += (y_val ** 2)
                    sum_diff[li, lj] += diff_sq
                    count[li, lj]    += 1

        # Clear cache
        torch.cuda.empty_cache()
        print(f"Processed year file: {ssrd_path.name}")
        print_memory_usage()

    # -------------------------------------------------------------------------
    # 7) Final overall RMSE and R² (global) if you still want them
    # -------------------------------------------------------------------------
    overall_rmse = np.sqrt(global_squared_error / global_count)
    sst = global_sum_y_squared - (global_sum_y**2) / global_count
    overall_r2 = 1 - global_squared_error / sst if sst > 0 else np.nan

    print("Overall RMSE:", overall_rmse)
    print("Overall R2:", overall_r2)
    print('Finished all years!')

    # -------------------------------------------------------------------------
    # 8) Compute pixel-wise RMSE and R²
    # -------------------------------------------------------------------------
    # SSE = sum(y - f)^2
    SSE = sum_diff
    # SST = sum(y^2) - (sum(y)^2 / n)
    SST = sum_y2 - (sum_y**2)/np.maximum(count, 1e-12)

    # We'll avoid divide-by-zero by masking out places with count=0 or SST=0
    valid_pix = (count > 0.5) & (SST > 1e-12)

    pixel_rmse = np.full((n_lat, n_lon), np.nan, dtype=np.float64)
    pixel_r2   = np.full((n_lat, n_lon), np.nan, dtype=np.float64)

    pixel_rmse[valid_pix] = np.sqrt( SSE[valid_pix] / count[valid_pix] )
    pixel_r2[valid_pix]   = 1.0 - (SSE[valid_pix] / SST[valid_pix])

    # -------------------------------------------------------------------------
    # 9) Save pixel-wise RMSE, R2 to NetCDF
    # -------------------------------------------------------------------------
    # Build an xarray Dataset
    ds_out = xr.Dataset(
        {
            "RMSE": (("latitude", "longitude"), pixel_rmse),
            "R2":   (("latitude", "longitude"), pixel_r2)
        },
        coords = {
            "latitude":  (("latitude",), lat_vals),
            "longitude": (("longitude",), lon_vals)
        }
    )

    # Example: save to "pixel_metrics.nc"
    ds_out.to_netcdf("./outputs/pixel_metrics_default_"+case+"_gpu.nc")
    print("Saved pixel-wise metrics to ./outputs/pixel_metrics_default_"+case+".nc")

# funtion to fine tune the pre-trained model
def fine_tuning(case='vpd3',pft_code=True):
    selected_pixels = xr.open_dataset("./outputs/selected_pixels_2012_2021_"+case+".nc")

    # Preprocess data
    if pft_code:
        selected_pixels['is_tropical'] = xr.where(selected_pixels['pft'] == PFT['tropical'], 1.0, 0.0)
        selected_pixels['is_temperate'] = xr.where(selected_pixels['pft'] == PFT['temperate'], 1.0, 0.0)
        selected_pixels['is_boreal'] = xr.where(selected_pixels['pft'] == PFT['boreal'], 1.0, 0.0)
        predictors = ['ssrd', 'is_tropical', 'is_temperate', 'is_boreal']    
    else:
        predictors = ['ssrd']
    target_gpp = 'gpp'
    X_train_full, y_train_full = preprocess_data(selected_pixels, predictors, target_gpp)

    # Split data into training and validation sets
    # X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_full, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_full, dtype=torch.float32).unsqueeze(1)  # Ensure correct shape
    # X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    # y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    # Create DataLoaders
    batch_size = 2**8  # You can adjust this value based on your available memory and GPU capacity
    train_loader = DataLoader(ClimateDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(ClimateDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    input_dim = len(predictors)  # Number of predictors
    model = NeuralNet(input_dim)
    if pft_code:
        model.load_state_dict(torch.load("./outputs/model_weights_pre_training_"+case+"_one-hot_encoding.pth"))
    else:
        model.load_state_dict(torch.load("./outputs/model_weights_pre_training_"+case+".pth"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()

    # Freeze Early Layers (fc1, fc2) to Preserve Pre-Trained Knowledge
    for param in model.fc1.parameters():
        param.requires_grad = False
    for param in model.fc2.parameters():
        param.requires_grad = False

    # Redefine Optimizer for Fine-Tuning (Only Updating fc3 and fc4)
    optimizer = optim.Adam(
        list(model.fc3.parameters()) + list(model.fc4.parameters()), lr=0.001
    )

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Validation Step
        # model.eval()
        # total_val_loss = 0
        # with torch.no_grad():
        #     for batch_X, batch_y in val_loader:
        #         batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        #         outputs = model(batch_X)
        #         loss = criterion(outputs, batch_y)
        #         total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        # avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.10f}")

        # Clear cache to free up memory
        torch.cuda.empty_cache()

    # Save the final model
    # Save the final model
    if pft_code:
        torch.save(model.state_dict(), "./outputs/model_weights_fine_tuning_"+case+"_one-hot_encoding.pth")
    else:
        torch.save(model.state_dict(), "./outputs/model_weights_fine_tuning_"+case+".pth")
    print('Fine tuning finished!')

# function to fine tune the pre-trained model
def fine_tuning_EVI_encode(case='vpd3',pft_code=True):
    selected_pixels = xr.open_dataset("./outputs/selected_pixels_2012_2021_"+case+".nc")

    # Preprocess data
    if pft_code:
        predictors = ['ssrd', 'evi']    
    else:
        predictors = ['ssrd']
    target_gpp = 'gpp'
    X_train_full, y_train_full = preprocess_data(selected_pixels, predictors, target_gpp)

    # Split data into training and validation sets
    # X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_full, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_full, dtype=torch.float32).unsqueeze(1)  # Ensure correct shape
    # X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    # y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    # Create DataLoaders
    batch_size = 2**8  # You can adjust this value based on your available memory and GPU capacity
    train_loader = DataLoader(ClimateDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(ClimateDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    input_dim = len(predictors)  # Number of predictors
    model = NeuralNet(input_dim)
    if pft_code:
        model.load_state_dict(torch.load("./outputs/model_weights_pre_training_"+case+".pth"))
    else:
        model.load_state_dict(torch.load("./outputs/model_weights_pre_training_"+case+".pth"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()

    # Freeze Early Layers (fc1, fc2) to Preserve Pre-Trained Knowledge
    for param in model.fc1.parameters():
        param.requires_grad = False
    for param in model.fc2.parameters():
        param.requires_grad = False

    # Redefine Optimizer for Fine-Tuning (Only Updating fc3 and fc4)
    optimizer = optim.Adam(
        list(model.fc3.parameters()) + list(model.fc4.parameters()), lr=0.001
    )

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Validation Step
        # model.eval()
        # total_val_loss = 0
        # with torch.no_grad():
        #     for batch_X, batch_y in val_loader:
        #         batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        #         outputs = model(batch_X)
        #         loss = criterion(outputs, batch_y)
        #         total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        # avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.10f}")

        # Clear cache to free up memory
        torch.cuda.empty_cache()

    # Save the final model
    # Save the final model
    if pft_code:
        torch.save(model.state_dict(), "./outputs/model_weights_fine_tuning_"+case+".pth")
    else:
        torch.save(model.state_dict(), "./outputs/model_weights_fine_tuning_"+case+".pth")
    print('Fine tuning finished!')

# funtion to fine tune the pre-trained model
def fine_tuning_EVI_encode(case='vpd3',pft_code=True):
    selected_pixels = xr.open_dataset("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/selected_pixels_2012_2021_pft_EVI.nc")

    evi_temperate_1 = np.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/evi_temperate_1.npy")

    # Create arrays for all four quarterly values
    # evi_arrays = []
    # for q in range(4):
        # Create an array filled with NaNs of the same shape as the original dataset
    evi_q_array = np.full((selected_pixels.dims['pixel'],selected_pixels.dims['time']), np.nan)
    
    # For all time steps, apply the quarterly value to each PFT region
    for t in range(selected_pixels.dims['time']):
        evi_q_array[:,t] = evi_temperate_1
    
        # evi_arrays.append(evi_q_array)

    # Add the new variables to the dataset
    # for q in range(4):
    selected_pixels['evi_1'] = (('pixel', 'time'), evi_q_array)

    # Initialize model, loss function, and optimizer
    evi_predictors = np.array([f"evi_{i}" for i in range(1, 2)])
    dynamic_predictors = ['ssrd']
    predictors = np.concatenate((dynamic_predictors, evi_predictors)) 
    
    # input_dim = len(predictors)  # Number of predictors
    dynamic_input_dim = len(dynamic_predictors)
    static_input_dim = len(evi_predictors)
    
    target_gpp = 'gpp'
    X_train_full, y_train_full = preprocess_data(selected_pixels, predictors, target_gpp)

    # Split data into training and validation sets
    # X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_full, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_full, dtype=torch.float32).unsqueeze(1)  # Ensure correct shape
    # X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    # y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    # Create DataLoaders
    batch_size = 2**8  # You can adjust this value based on your available memory and GPU capacity
    train_loader = DataLoader(ClimateDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(ClimateDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    input_dim = dynamic_input_dim+static_input_dim  # Number of predictors
    model = NeuralNet(input_dim)
    model.load_state_dict(torch.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/model_weights_pre_training_"+case+".pth"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()

    # Freeze Early Layers (fc1, fc2) to Preserve Pre-Trained Knowledge
    for param in model.fc1.parameters():
        param.requires_grad = False
    for param in model.fc2.parameters():
        param.requires_grad = False

    # Redefine Optimizer for Fine-Tuning (Only Updating fc3 and fc4)
    optimizer = optim.Adam(
        list(model.fc3.parameters()) + list(model.fc4.parameters()), lr=0.001
    )

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Validation Step
        # model.eval()
        # total_val_loss = 0
        # with torch.no_grad():
        #     for batch_X, batch_y in val_loader:
        #         batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        #         outputs = model(batch_X)
        #         loss = criterion(outputs, batch_y)
        #         total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        # avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.10f}")

        # Clear cache to free up memory
        torch.cuda.empty_cache()

    # Save the final model
    # Save the final model
    torch.save(model.state_dict(), "./outputs/model_weights_fine_tuning_"+case+".pth")
    
    print('Fine tuning finished!')

# function to forward run the fine-tuned model
def fine_tuning_forward(start_year=2012, end_year=2021,case='vpd3'):
    # Load the global mean and std for standardization
    ds_stats = xr.open_dataset("./outputs/combined_stats_land_"+case+".nc")

    # Load the selected pixel coordinates
    selected_longitudes = np.load("./outputs/selected_longitudes_"+case+".npy")
    selected_latitudes = np.load("./outputs/selected_latitudes_"+case+".npy")
    selected_coords = np.column_stack((selected_longitudes, selected_latitudes))

    root = Path("/Net/Groups/data_BGC/era5/e1/0d25_daily")
    ssrd_paths = sorted(
        (root / f"ssrd/ssrd.daily.fc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )
    vpd_paths = sorted(
        (root / f"vpd_cf/vpd_cf.daily.calc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )

    # Initialize model and load pre-training weights
    input_dim = 2  # predictors: ssrd and vpd
    model = NeuralNet(input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load("./outputs/model_weights_fine_tuning_"+case+".pth"))
    model.eval()

    predictors = ['ssrd', 'vpd']
    target_gpp = 'gpp'

    # Initialize global accumulators
    global_squared_error = 0.0
    global_count = 0
    global_sum_y = 0.0
    global_sum_y_squared = 0.0

    # Loop over each year's files
    for ssrd_path, vpd_path in zip(ssrd_paths, vpd_paths):
        # Load one file at a time and merge datasets
        ds_ssrd = xr.open_dataset(ssrd_path)
        ds_ssrd = ds_ssrd * 11.574
        ds_vpd = xr.open_dataset(vpd_path)
        ds_vpd = ds_vpd.rename({"vpd_cf": "vpd"})
        ds = xr.merge([ds_ssrd, ds_vpd])
        
        # Calculate GPP and SIF
        ds = SIF_GPP_model(ds)
        # uncertainty  
        ds = SIF_GPP_model_uncertainty(ds)

        # Standardize ssrd and vpd using global stats
        ds["ssrd"] = (ds["ssrd"] - ds_stats['SW_IN_mean'].values) / ds_stats['SW_IN_std'].values
        ds["vpd"] = (ds["vpd"] - ds_stats['VPD_mean'].values) / ds_stats['VPD_std'].values
        ds['gpp'] = (ds['gpp'] - ds_stats['GPP_mean'].values) / ds_stats['GPP_std'].values

        # Extract coordinate values
        lon = ds.longitude.values   # shape (1440,)
        lat = ds.latitude.values     # shape (720,)

        # Create 2D meshgrid for lon and lat (note the indexing='ij' to match dims order)
        lon2d, lat2d = np.meshgrid(lon, lat, indexing='ij')  # both shape (1440, 720)
        print(ds.dims)
        # Initialize a boolean mask (True where you want to remove the data)
        mask = np.zeros(lon2d.shape, dtype=bool)

        # Loop over each selected coordinate pair and update the mask.
        # Use np.isclose if the coordinates are floats to avoid precision issues.
        for selected in selected_coords:
            sel_lon, sel_lat = selected
            mask |= (np.isclose(lon2d, sel_lon)) & (np.isclose(lat2d, sel_lat))

        # Convert the mask into an xarray DataArray, associating the correct coords and dims.
        mask_da = xr.DataArray(mask, dims=["longitude", "latitude"],
                            coords={"longitude": lon, "latitude": lat})

        # Remove selected pixels using where (drop=True)
        ds_remaining = ds.where(~mask_da, drop=True)

        # Preprocess and get coordinates
        X_data, y_data = preprocess_data(ds_remaining, predictors, target_gpp)
        
        dataset_climate = ClimateDataset(X_data, y_data)
        dataloader = DataLoader(dataset_climate, batch_size=64, shuffle=False)
        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                # Squeeze outputs to remove the extra dimension (shape becomes (batch_size,))
                outputs_np = outputs.cpu().numpy().squeeze()
                batch_y_np = batch_y.numpy()
                
                # Now both outputs_np and batch_y_np have the same shape
                valid = ~np.isnan(outputs_np) & ~np.isnan(batch_y_np)
                outputs_valid = outputs_np[valid]
                batch_y_valid = batch_y_np[valid]
                
                # Update global accumulators with valid values
                diff = outputs_valid - batch_y_valid
                global_squared_error += np.sum(diff**2)
                n = batch_y_valid.size
                global_count += n
                global_sum_y += np.sum(batch_y_valid)
                global_sum_y_squared += np.sum(batch_y_valid**2)
        
        # Clear cache to free up memory
        torch.cuda.empty_cache()
        print(f"Processed year file: {ssrd_path.name}")
        print_memory_usage()

    # Compute overall RMSE and R2 using global accumulators
    overall_rmse = np.sqrt(global_squared_error / global_count)
    # global_mean_y = global_sum_y / global_count
    sst = global_sum_y_squared - (global_sum_y**2) / global_count
    overall_r2 = 1 - global_squared_error / sst
    print("Overall RMSE:", overall_rmse)
    print("Overall R2:", overall_r2)

    print('Fine-tuning forward run finished!')

def fine_tuning_forward_pixelwise(start_year=2012, end_year=2013,case='vpd3',use_f_ssrd=False):
    # -------------------------------------------------------------------------
    # 1) Load global mean/std for standardization
    # -------------------------------------------------------------------------
    ds_stats = xr.open_dataset("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/combined_stats_land_"+case+".nc")
    GPP_mean = ds_stats['GPP_mean'].values.item()
    GPP_std  = ds_stats['GPP_std'].values.item()

    SW_IN_mean = ds_stats['SW_IN_mean'].values.item()
    SW_IN_std  = ds_stats['SW_IN_std'].values.item()
    VPD_mean   = ds_stats['VPD_mean'].values.item()
    VPD_std    = ds_stats['VPD_std'].values.item()

    # -------------------------------------------------------------------------
    # 2) Load the selected pixel coordinates to *exclude* from evaluation
    # -------------------------------------------------------------------------
    selected_longitudes = np.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/selected_longitudes_"+case+".npy")
    selected_latitudes  = np.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/selected_latitudes_"+case+".npy")
    selected_coords     = np.column_stack((selected_longitudes, selected_latitudes))

    # -------------------------------------------------------------------------
    # 3) Prepare ERA5 file paths for each year
    # -------------------------------------------------------------------------
    root = Path("/Net/Groups/data_BGC/era5/e1/0d25_daily")
    ssrd_paths = sorted(
        (root / f"ssrd/ssrd.daily.fc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )
    vpd_paths = sorted(
        (root / f"vpd_cf/vpd_cf.daily.calc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )

    # -------------------------------------------------------------------------
    # 4) Initialize your neural net and load weights
    # -------------------------------------------------------------------------
    input_dim = 2  # we have 2 predictors: ssrd and vpd
    model = NeuralNet(input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load("./outputs/model_weights_fine_tuning_"+case+".pth"))
    model.eval()

    predictors = ['ssrd', 'vpd']
    target_gpp = 'gpp'

    # -------------------------------------------------------------------------
    # 5) Prepare large aggregator arrays for pixel-wise statistics
    #    We'll assume the domain is 720 lat × 1440 lon (as per your data).
    #    We will fill them incrementally over time.
    # -------------------------------------------------------------------------
    # We won't know the *exact* shape until we open at least one file and check
    # the coordinate dimension lengths. For safety:
    sample_ds = xr.open_dataset(ssrd_paths[0])
    lat_vals = sample_ds.latitude.values  # shape ~ (720,)
    lon_vals = sample_ds.longitude.values # shape ~ (1440,)
    n_lat = lat_vals.size
    n_lon = lon_vals.size
    
    # Each aggregator is 2D: [n_lat, n_lon]
    sum_y   = np.zeros((n_lat, n_lon), dtype=np.float64)
    sum_y2  = np.zeros((n_lat, n_lon), dtype=np.float64)
    sum_diff   = np.zeros((n_lat, n_lon), dtype=np.float64)
    count   = np.zeros((n_lat, n_lon), dtype=np.float64)

    # -------------------------------------------------------------------------
    # 6) Loop over each year, build dataset, run model predictions in batches
    # -------------------------------------------------------------------------
    # (We will still keep track of global accumulators to see overall RMSE, R2)
    global_squared_error = 0.0
    global_count = 0
    global_sum_y = 0.0
    global_sum_y_squared = 0.0

    ssrd_path = ssrd_paths[0]
    vpd_path = vpd_paths[0]
    for ssrd_path, vpd_path in zip(ssrd_paths, vpd_paths):
        # ---- Load & rename
        ds_ssrd = xr.open_dataset(ssrd_path) * 11.574
        ds_vpd = xr.open_dataset(vpd_path).rename({"vpd_cf": "vpd"})
        ds = xr.merge([ds_ssrd, ds_vpd])

        # ---- Compute GPP and SIF, plus uncertainties
        ds = SIF_GPP_model(ds,use_f_ssrd)
        ds = SIF_GPP_model_uncertainty(ds)

        # ---- Standardize the input and the target
        ds["ssrd"] = (ds["ssrd"] - SW_IN_mean) / SW_IN_std
        ds["vpd"]  = (ds["vpd"]  - VPD_mean)   / VPD_std
        ds['gpp']  = (ds['gpp']  - GPP_mean)   / GPP_std

        # ---- Exclude your selected coordinates
        lon_1d = ds.longitude.values  # shape (1440,)
        lat_1d = ds.latitude.values   # shape (720,)
        lon2d, lat2d = np.meshgrid(lon_1d, lat_1d, indexing='ij')  # (1440, 720)
        
        mask = np.zeros(lon2d.shape, dtype=bool)
        for sel_lon, sel_lat in selected_coords:
            mask |= (np.isclose(lon2d, sel_lon)) & (np.isclose(lat2d, sel_lat))
        # Convert to DataArray
        mask_da = xr.DataArray(mask, dims=["longitude", "latitude"],
                               coords={"longitude": lon_1d, "latitude": lat_1d})

        ds_remaining = ds.where(~mask_da, drop=True)

        # ---- Flatten data for PyTorch
        # We'll retrieve not just X_data, y_data but also the lat/lon indices.
        X_data, y_data, lat_idx_arr, lon_idx_arr = preprocess_data_lat_lon(
            ds_remaining, predictors, target_gpp
        )

        dataset_climate = ClimateDataset_lat_lon(X_data, y_data, lat_idx_arr, lon_idx_arr)
        dataloader = DataLoader(dataset_climate, batch_size=256, shuffle=False)

        with torch.no_grad():
            for batch_X, batch_y, lat_batch, lon_batch in dataloader:
                outputs = model(batch_X.to(device))
                # Bring predictions and targets back to CPU
                outputs_np = outputs.cpu().detach().numpy().squeeze()
                batch_y_np = batch_y.numpy()
                # Invert the standardization *if* you want metrics in original GPP units
                outputs_np_orig = outputs_np * GPP_std + GPP_mean
                batch_y_np_orig = batch_y_np * GPP_std + GPP_mean

                # Valid (non-NaN) mask
                valid_mask = ~np.isnan(outputs_np_orig) & ~np.isnan(batch_y_np_orig)
                outputs_valid = outputs_np_orig[valid_mask]
                batch_y_valid = batch_y_np_orig[valid_mask]
                lat_idx_valid = lat_batch[valid_mask]
                lon_idx_valid = lon_batch[valid_mask]

                # ---- Update global accumulators
                diff = outputs_valid - batch_y_valid
                global_squared_error += np.sum(diff**2)
                n_valid = batch_y_valid.size
                global_count += n_valid
                global_sum_y += np.sum(batch_y_valid)
                global_sum_y_squared += np.sum(batch_y_valid**2)

                # ---- Update pixel-wise accumulators
                # Ensure lat_idx_valid and lon_idx_valid are numpy arrays
                if torch.is_tensor(lat_idx_valid):
                    lat_idx_valid = lat_idx_valid.cpu().numpy()
                if torch.is_tensor(lon_idx_valid):
                    lon_idx_valid = lon_idx_valid.cpu().numpy()

                lat_indices = np.argmin(np.abs(lat_vals[:, None] - lat_idx_valid[None, :]), axis=0)
                lon_indices = np.argmin(np.abs(lon_vals[:, None] - lon_idx_valid[None, :]), axis=0)

                # Update pixel-wise accumulators with vectorized operations
                np.add.at(sum_y,   (lat_indices, lon_indices), batch_y_valid)
                np.add.at(sum_y2,  (lat_indices, lon_indices), batch_y_valid**2)
                np.add.at(sum_diff,(lat_indices, lon_indices), (batch_y_valid - outputs_valid)**2)
                np.add.at(count,   (lat_indices, lon_indices), 1)

        # Clear cache
        torch.cuda.empty_cache()
        print(f"Processed year file: {ssrd_path.name}")
        print_memory_usage()

    # -------------------------------------------------------------------------
    # 7) Final overall RMSE and R² (global) if you still want them
    # -------------------------------------------------------------------------
    overall_rmse = np.sqrt(global_squared_error / global_count)
    sst = global_sum_y_squared - (global_sum_y**2) / global_count
    overall_r2 = 1 - global_squared_error / sst if sst > 0 else np.nan

    print("Overall RMSE:", overall_rmse)
    print("Overall R2:", overall_r2)
    print('Finished all years!')

    # -------------------------------------------------------------------------
    # 8) Compute pixel-wise RMSE and R²
    # -------------------------------------------------------------------------
    # SSE = sum(y - f)^2
    SSE = sum_diff
    # SST = sum(y^2) - (sum(y)^2 / n)
    SST = sum_y2 - (sum_y**2)/np.maximum(count, 1e-12)

    # We'll avoid divide-by-zero by masking out places with count=0 or SST=0
    valid_pix = (count > 0.5) & (SST > 1e-12)

    pixel_rmse = np.full((n_lat, n_lon), np.nan, dtype=np.float64)
    pixel_r2   = np.full((n_lat, n_lon), np.nan, dtype=np.float64)

    pixel_rmse[valid_pix] = np.sqrt( SSE[valid_pix] / count[valid_pix] )
    pixel_r2[valid_pix]   = 1.0 - (SSE[valid_pix] / SST[valid_pix])

    # -------------------------------------------------------------------------
    # 9) Save pixel-wise RMSE, R2 to NetCDF
    # -------------------------------------------------------------------------
    # Build an xarray Dataset
    ds_out = xr.Dataset(
        {
            "RMSE": (("latitude", "longitude"), pixel_rmse),
            "R2":   (("latitude", "longitude"), pixel_r2)
        },
        coords = {
            "latitude":  (("latitude",), lat_vals),
            "longitude": (("longitude",), lon_vals)
        }
    )

    # Example: save to "pixel_metrics.nc"
    ds_out.to_netcdf("./outputs/pixel_metrics_fine_tuning_"+case+".nc")
    print("Saved pixel-wise metrics to ./outputs/pixel_metrics_fine_tuning_"+case+".nc")

def fine_tuning_forward_subsample(start_year=2012, end_year=2013,case='vpd3',
                                  sample_size=1000,use_f_ssrd=False,pft_code=True):
    # -------------------------------------------------------------------------
    # 1) Load global mean/std for standardization
    # -------------------------------------------------------------------------
    ds_stats = xr.open_dataset("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/combined_stats_land_"+case+".nc")
    GPP_mean = ds_stats['GPP_mean'].values.item()
    GPP_std  = ds_stats['GPP_std'].values.item()

    SW_IN_mean = ds_stats['SW_IN_mean'].values.item()
    SW_IN_std  = ds_stats['SW_IN_std'].values.item()
    VPD_mean   = ds_stats['VPD_mean'].values.item()
    VPD_std    = ds_stats['VPD_std'].values.item()

    # -------------------------------------------------------------------------
    # 2) Load the selected pixel coordinates to *exclude* from evaluation
    # -------------------------------------------------------------------------
    selected_longitudes = np.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/selected_longitudes_"+case+".npy")
    selected_latitudes  = np.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/selected_latitudes_"+case+".npy")
    selected_coords     = np.column_stack((selected_longitudes, selected_latitudes))

    # -------------------------------------------------------------------------
    # 3) Prepare ERA5 file paths for each year
    # -------------------------------------------------------------------------
    root = Path("/Net/Groups/data_BGC/era5/e1/0d25_daily")
    ssrd_paths = sorted(
        (root / f"ssrd/ssrd.daily.fc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )
    vpd_paths = sorted(
        (root / f"vpd_cf/vpd_cf.daily.calc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )

    """
    subsample the data to increase the training speed
    """
    if use_f_ssrd:
        ssrd_path = ssrd_paths[0]
        ds_temp = xr.open_dataset(ssrd_path)
        mask_da = xr.open_dataarray("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/subsample_mask_forward_"+case+".nc")
    else:
        ssrd_path = ssrd_paths[0]
        ds_temp = xr.open_dataset(ssrd_path)
        # land mask
        land_mask = create_land_mask(ds_temp)
        ds_temp = ds_temp.where(land_mask)

        #eddy covariance mask
        EC_mask = subsampling(selected_coords,ds_temp)
        ds_temp = ds_temp.where(~EC_mask)

        valid_mask = ~np.isnan(ds_temp['ssrd'].isel(time=0))
        valid_pixels = np.argwhere(valid_mask.values)
        num_selected = min(sample_size, len(valid_pixels))
        selected_indices = np.random.choice(valid_pixels.shape[0], num_selected, replace=False)
        selected_positions = valid_pixels[selected_indices,:] 
        #Save longitude and latitude of selected pixels.
        # Assuming overall_mean has 1D latitude and longitude coordinates with dimensions "latitude" then "longitude".
        selected_lats = ds_temp.latitude.values[selected_positions[:,0]]
        selected_lons = ds_temp.longitude.values[selected_positions[:,1]]
        sample_coords = np.column_stack((selected_lats, selected_lons))
        mask_da = subsampling(sample_coords,ds_temp)

        mask_da.to_netcdf("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/subsample_mask_forward_"+case+".nc")
    # -------------------------------------------------------------------------
    # 4) Initialize the neural net and load weights
    # -------------------------------------------------------------------------
    if pft_code:
        predictors = ['ssrd','is_tropical','is_temperate','is_boreal']
    else:
        predictors = ['ssrd']
    input_dim = len(predictors)  # we have 2 predictors: ssrd and vpd
    model = NeuralNet(input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if pft_code:
        model.load_state_dict(torch.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/model_weights_fine_tuning_"+case+"_one-hot_encoding.pth"))
    else:
        model.load_state_dict(torch.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/model_weights_fine_tuning_"+case+".pth"))

    model.eval()

    target_gpp = 'gpp'

    # -------------------------------------------------------------------------
    # 5) Prepare large aggregator arrays for pixel-wise statistics
    # -------------------------------------------------------------------------
    # We won't know the *exact* shape until we open at least one file and check
    # the coordinate dimension lengths. For safety:
    lat_vals = ds_temp.latitude.values  # shape ~ (720,)
    lon_vals = ds_temp.longitude.values # shape ~ (1440,)
    
    # ssrd_path = ssrd_paths[0]
    # vpd_path = vpd_paths[0]
    years_range = np.arange(start_year, end_year + 1)
    for year_idx, ssrd_path, vpd_path in zip(years_range, ssrd_paths, vpd_paths):
        # ---- Load & rename
        ds_ssrd = xr.open_dataset(ssrd_path) * 11.574
        ds_vpd = xr.open_dataset(vpd_path).rename({"vpd_cf": "vpd"})
        ds = xr.merge([ds_ssrd, ds_vpd])

        # ---- Compute GPP and SIF, plus uncertainties
        ds = SIF_GPP_model_ssrd(ds,use_f_ssrd)
        ds = SIF_GPP_model_ssrd_uncertainty(ds)

        # ---- Standardize the input and the target
        ds["ssrd"] = (ds["ssrd"] - SW_IN_mean) / SW_IN_std
        ds["vpd"]  = (ds["vpd"]  - VPD_mean)   / VPD_std
        ds['gpp']  = (ds['gpp']  - GPP_mean)   / GPP_std

        if pft_code:
            ds['is_tropical'] = xr.where(ds['pft'] == PFT['tropical'], 1.0, 0.0)
            ds['is_temperate'] = xr.where(ds['pft'] == PFT['temperate'], 1.0, 0.0)
            ds['is_boreal'] = xr.where(ds['pft'] == PFT['boreal'], 1.0, 0.0)
        #subsample the data
        ds_remaining = ds.where(mask_da)

        # ---- Flatten data for PyTorch
        # We'll retrieve not just X_data, y_data but also the lat/lon indices.
        X_data, y_data, lat_idx_arr, lon_idx_arr, time_idx_arr = preprocess_data_lat_lon(
            ds_remaining, predictors, target_gpp
        )

        batch_size = 2**8
        dataset_climate = ClimateDataset_lat_lon(X_data, y_data, lat_idx_arr, lon_idx_arr, time_idx_arr)
        dataloader = DataLoader(dataset_climate, batch_size=batch_size, shuffle=False)

        pred_gpp_array = np.full((ds.dims["time"], ds.dims["latitude"], ds.dims["longitude"]), np.nan)

        with torch.no_grad():
            for batch_X, batch_y, lat_batch, lon_batch, time_batch in dataloader:
                outputs = model(batch_X.float().to(device))
                # Bring predictions and targets back to CPU
                outputs_np = outputs.cpu().numpy().squeeze()
                batch_y_np = batch_y.numpy()
                # Invert the standardization if you want metrics in original GPP units
                outputs_np_orig = outputs_np * GPP_std + GPP_mean
                batch_y_np_orig = batch_y_np * GPP_std + GPP_mean

                # Valid (non-NaN) mask
                valid_mask = ~np.isnan(outputs_np_orig) & ~np.isnan(batch_y_np_orig)
                outputs_valid = outputs_np_orig[valid_mask]
                lat_idx_valid = lat_batch[valid_mask]
                lon_idx_valid = lon_batch[valid_mask]
                time_idx_valid = time_batch[valid_mask]

                # Ensure lat_idx_valid and lon_idx_valid are numpy arrays
                if torch.is_tensor(lat_idx_valid):
                    lat_idx_valid = lat_idx_valid.cpu().numpy()
                if torch.is_tensor(lon_idx_valid):
                    lon_idx_valid = lon_idx_valid.cpu().numpy()

                lat_indices = np.array([np.where(lat_vals == val)[0][0] for val in lat_idx_valid])
                lon_indices = np.array([np.where(lon_vals == val)[0][0] for val in lon_idx_valid])

                # Assign predictions to the correct positions in pred_gpp_array
                pred_gpp_array[time_idx_valid, lat_indices, lon_indices] = outputs_valid

        ds_remaining['gpp_hat'] = (("time","latitude","longitude"),pred_gpp_array)
        if pft_code:
            ds_remaining.to_netcdf("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/fine_tuning_forward_output_"+str(year_idx)+"_"+case+"_one-hot_encoding_subsample.nc")
        else:
            ds_remaining.to_netcdf("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/fine_tuning_forward_output_"+str(year_idx)+"_"+case+"_subsample.nc")
        # Clear cache
        torch.cuda.empty_cache()
        print(f"Processed year file: {ssrd_path.name}")
        print_memory_usage()

def fine_tuning_forward_subsample_EVI(start_year=2012, end_year=2013,case='vpd3',
                                  sample_size=1000,use_f_ssrd=False,pft_code=True):
    # -------------------------------------------------------------------------
    # 1) Load global mean/std for standardization
    # -------------------------------------------------------------------------
    ds_stats = xr.open_dataset("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/combined_stats_land_"+case+".nc")
    GPP_mean = ds_stats['GPP_mean'].values.item()
    GPP_std  = ds_stats['GPP_std'].values.item()

    SW_IN_mean = ds_stats['SW_IN_mean'].values.item()
    SW_IN_std  = ds_stats['SW_IN_std'].values.item()
    VPD_mean   = ds_stats['VPD_mean'].values.item()
    VPD_std    = ds_stats['VPD_std'].values.item()

    EVI_mean   = ds_stats['EVI_mean'].values.item()
    EVI_std    = ds_stats['EVI_std'].values.item()

    # -------------------------------------------------------------------------
    # 2) Load the selected pixel coordinates to *exclude* from evaluation
    # -------------------------------------------------------------------------
    selected_longitudes = np.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/selected_longitudes_pft_best.npy")
    selected_latitudes  = np.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/selected_latitudes_pft_best.npy")
    selected_coords     = np.column_stack((selected_longitudes, selected_latitudes))

    # -------------------------------------------------------------------------
    # 3) Prepare ERA5 file paths for each year
    # -------------------------------------------------------------------------
    root = Path("/Net/Groups/data_BGC/era5/e1/0d25_daily")
    ssrd_paths = sorted(
        (root / f"ssrd/ssrd.daily.fc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )
    vpd_paths = sorted(
        (root / f"vpd_cf/vpd_cf.daily.calc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )

    """
    subsample the data to increase the training speed
    """
    if use_f_ssrd:
        ssrd_path = ssrd_paths[0]
        ds_temp = xr.open_dataset(ssrd_path)
        mask_da = xr.open_dataarray("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/subsample_mask_forward_pft_best.nc")
    else:
        ssrd_path = ssrd_paths[0]
        ds_temp = xr.open_dataset(ssrd_path)
        # land mask
        land_mask = create_land_mask(ds_temp)
        ds_temp = ds_temp.where(land_mask)

        #eddy covariance mask
        EC_mask = subsampling(selected_coords,ds_temp)
        ds_temp = ds_temp.where(~EC_mask)

        valid_mask = ~np.isnan(ds_temp['ssrd'].isel(time=0))
        valid_pixels = np.argwhere(valid_mask.values)
        num_selected = min(sample_size, len(valid_pixels))
        selected_indices = np.random.choice(valid_pixels.shape[0], num_selected, replace=False)
        selected_positions = valid_pixels[selected_indices,:] 
        #Save longitude and latitude of selected pixels.
        # Assuming overall_mean has 1D latitude and longitude coordinates with dimensions "latitude" then "longitude".
        selected_lats = ds_temp.latitude.values[selected_positions[:,0]]
        selected_lons = ds_temp.longitude.values[selected_positions[:,1]]
        sample_coords = np.column_stack((selected_lats, selected_lons))
        mask_da = subsampling(sample_coords,ds_temp)

        mask_da.to_netcdf("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/subsample_mask_forward_"+case+".nc")
    # -------------------------------------------------------------------------
    # 4) Initialize the neural net and load weights
    # -------------------------------------------------------------------------
    if pft_code:
        predictors = ['ssrd','evi']
    else:
        predictors = ['ssrd']
    input_dim = len(predictors)  # we have 2 predictors: ssrd and vpd
    model = NeuralNet(input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if pft_code:
        model.load_state_dict(torch.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/model_weights_fine_tuning_"+case+".pth"))
    else:
        model.load_state_dict(torch.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/model_weights_fine_tuning_"+case+".pth"))

    model.eval()

    target_gpp = 'gpp'

    # -------------------------------------------------------------------------
    # 5) Prepare large aggregator arrays for pixel-wise statistics
    # -------------------------------------------------------------------------
    # We won't know the *exact* shape until we open at least one file and check
    # the coordinate dimension lengths. For safety:
    lat_vals = ds_temp.latitude.values  # shape ~ (720,)
    lon_vals = ds_temp.longitude.values # shape ~ (1440,)
    
    # ssrd_path = ssrd_paths[0]
    # vpd_path = vpd_paths[0]
    years_range = np.arange(start_year, end_year + 1)
    for year_idx, ssrd_path, vpd_path in zip(years_range, ssrd_paths, vpd_paths):
        # ---- Load & rename
        ds_ssrd = xr.open_dataset(ssrd_path) * 11.574
        ds_vpd = xr.open_dataset(vpd_path).rename({"vpd_cf": "vpd"})
        ds = xr.merge([ds_ssrd, ds_vpd])

        # ---- Compute GPP and SIF, plus uncertainties
        ds = SIF_GPP_EVI_model_ssrd(ds,use_f_ssrd)
        ds = SIF_GPP_EVI_model_ssrd_uncertainty(ds)

        # ---- Standardize the input and the target
        ds["ssrd"] = (ds["ssrd"] - SW_IN_mean) / SW_IN_std
        ds["vpd"]  = (ds["vpd"]  - VPD_mean)   / VPD_std
        ds['gpp']  = (ds['gpp']  - GPP_mean)   / GPP_std
        ds['evi'] =  (ds['evi'] - EVI_mean) / EVI_std
        # if pft_code:
        #     ds['is_tropical'] = xr.where(ds['pft'] == PFT['tropical'], 1.0, 0.0)
        #     ds['is_temperate'] = xr.where(ds['pft'] == PFT['temperate'], 1.0, 0.0)
        #     ds['is_boreal'] = xr.where(ds['pft'] == PFT['boreal'], 1.0, 0.0)
        #subsample the data
        ds_remaining = ds.where(mask_da)

        # ---- Flatten data for PyTorch
        # We'll retrieve not just X_data, y_data but also the lat/lon indices.
        X_data, y_data, lat_idx_arr, lon_idx_arr, time_idx_arr = preprocess_data_lat_lon(
            ds_remaining, predictors, target_gpp
        )

        batch_size = 2**8
        dataset_climate = ClimateDataset_lat_lon(X_data, y_data, lat_idx_arr, lon_idx_arr, time_idx_arr)
        dataloader = DataLoader(dataset_climate, batch_size=batch_size, shuffle=False)

        pred_gpp_array = np.full((ds.dims["time"], ds.dims["latitude"], ds.dims["longitude"]), np.nan)

        with torch.no_grad():
            for batch_X, batch_y, lat_batch, lon_batch, time_batch in dataloader:
                outputs = model(batch_X.float().to(device))
                # Bring predictions and targets back to CPU
                outputs_np = outputs.cpu().numpy().squeeze()
                batch_y_np = batch_y.numpy()
                # Invert the standardization if you want metrics in original GPP units
                outputs_np_orig = outputs_np * GPP_std + GPP_mean
                batch_y_np_orig = batch_y_np * GPP_std + GPP_mean

                # Valid (non-NaN) mask
                valid_mask = ~np.isnan(outputs_np_orig) & ~np.isnan(batch_y_np_orig)
                outputs_valid = outputs_np_orig[valid_mask]
                lat_idx_valid = lat_batch[valid_mask]
                lon_idx_valid = lon_batch[valid_mask]
                time_idx_valid = time_batch[valid_mask]

                # Ensure lat_idx_valid and lon_idx_valid are numpy arrays
                if torch.is_tensor(lat_idx_valid):
                    lat_idx_valid = lat_idx_valid.cpu().numpy()
                if torch.is_tensor(lon_idx_valid):
                    lon_idx_valid = lon_idx_valid.cpu().numpy()

                lat_indices = np.array([np.where(lat_vals == val)[0][0] for val in lat_idx_valid])
                lon_indices = np.array([np.where(lon_vals == val)[0][0] for val in lon_idx_valid])

                # Assign predictions to the correct positions in pred_gpp_array
                pred_gpp_array[time_idx_valid, lat_indices, lon_indices] = outputs_valid

        ds_remaining['gpp_hat'] = (("time","latitude","longitude"),pred_gpp_array)
        if pft_code:
            ds_remaining.to_netcdf("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/fine_tuning_forward_output_"+str(year_idx)+"_"+case+"_subsample.nc")
        else:
            ds_remaining.to_netcdf("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/fine_tuning_forward_output_"+str(year_idx)+"_"+case+"_subsample.nc")
        # Clear cache
        torch.cuda.empty_cache()
        print(f"Processed year file: {ssrd_path.name}")
        print_memory_usage()

def fine_tuning_forward_subsample_EVI_encode(start_year=2012, end_year=2013,case='vpd3',
                                  sample_size=1000,use_f_ssrd=False,pft_code=True):
    # -------------------------------------------------------------------------
    # 1) Load global mean/std for standardization
    # -------------------------------------------------------------------------
    ds_stats = xr.open_dataset("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/combined_stats_land_pft_EVI.nc")
    GPP_mean = ds_stats['GPP_mean'].values.item()
    GPP_std  = ds_stats['GPP_std'].values.item()

    SW_IN_mean = ds_stats['SW_IN_mean'].values.item()
    SW_IN_std  = ds_stats['SW_IN_std'].values.item()
    VPD_mean   = ds_stats['VPD_mean'].values.item()
    VPD_std    = ds_stats['VPD_std'].values.item()

    EVI_mean   = ds_stats['EVI_mean'].values.item()
    EVI_std    = ds_stats['EVI_std'].values.item()

    # -------------------------------------------------------------------------
    # 2) Load the selected pixel coordinates to *exclude* from evaluation
    # -------------------------------------------------------------------------
    selected_longitudes = np.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/selected_longitudes_pft_best.npy")
    selected_latitudes  = np.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/selected_latitudes_pft_best.npy")
    selected_coords     = np.column_stack((selected_longitudes, selected_latitudes))

    # -------------------------------------------------------------------------
    # 3) Prepare ERA5 file paths for each year
    # -------------------------------------------------------------------------
    root = Path("/Net/Groups/data_BGC/era5/e1/0d25_daily")
    ssrd_paths = sorted(
        (root / f"ssrd/ssrd.daily.fc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )
    vpd_paths = sorted(
        (root / f"vpd_cf/vpd_cf.daily.calc.era5.1440.720.{year}.nc")
        for year in range(start_year, end_year + 1)
    )

    """
    subsample the data to increase the training speed
    """
    if use_f_ssrd:
        ssrd_path = ssrd_paths[0]
        ds_temp = xr.open_dataset(ssrd_path)
        mask_da = xr.open_dataarray("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/subsample_mask_forward_pft_best.nc")
    else:
        ssrd_path = ssrd_paths[0]
        ds_temp = xr.open_dataset(ssrd_path)
        # land mask
        land_mask = create_land_mask(ds_temp)
        ds_temp = ds_temp.where(land_mask)

        #eddy covariance mask
        EC_mask = subsampling(selected_coords,ds_temp)
        ds_temp = ds_temp.where(~EC_mask)

        valid_mask = ~np.isnan(ds_temp['ssrd'].isel(time=0))
        valid_pixels = np.argwhere(valid_mask.values)
        num_selected = min(sample_size, len(valid_pixels))
        selected_indices = np.random.choice(valid_pixels.shape[0], num_selected, replace=False)
        selected_positions = valid_pixels[selected_indices,:] 
        #Save longitude and latitude of selected pixels.
        # Assuming overall_mean has 1D latitude and longitude coordinates with dimensions "latitude" then "longitude".
        selected_lats = ds_temp.latitude.values[selected_positions[:,0]]
        selected_lons = ds_temp.longitude.values[selected_positions[:,1]]
        sample_coords = np.column_stack((selected_lats, selected_lons))
        mask_da = subsampling(sample_coords,ds_temp)

        mask_da.to_netcdf("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/subsample_mask_forward_"+case+".nc")
    # -------------------------------------------------------------------------
    # 4) Initialize the neural net and load weights
    # -------------------------------------------------------------------------
    if pft_code:
        evi_predictors = np.array([f"evi_{i}" for i in range(1, 2)])
        dynamic_predictors = ['ssrd']
        predictors = np.concatenate((dynamic_predictors, evi_predictors)) 
    else:
        predictors = ['ssrd']
    # input_dim = len(predictors)  # Number of predictors
    dynamic_input_dim = len(dynamic_predictors)
    static_input_dim = len(evi_predictors)
    model = NeuralNet(dynamic_input_dim+static_input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if pft_code:
        model.load_state_dict(torch.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/model_weights_fine_tuning_"+case+".pth"))
    else:
        model.load_state_dict(torch.load("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/model_weights_fine_tuning_"+case+".pth"))

    model.eval()

    target_gpp = 'gpp'

    # -------------------------------------------------------------------------
    # 5) Prepare large aggregator arrays for pixel-wise statistics
    # -------------------------------------------------------------------------
    # We won't know the *exact* shape until we open at least one file and check
    # the coordinate dimension lengths. For safety:
    lat_vals = ds_temp.latitude.values  # shape ~ (720,)
    lon_vals = ds_temp.longitude.values # shape ~ (1440,)
    
    # ssrd_path = ssrd_paths[0]
    # vpd_path = vpd_paths[0]
    years_range = np.arange(start_year, end_year + 1)
    for year_idx, ssrd_path, vpd_path in zip(years_range, ssrd_paths, vpd_paths):
        # ---- Load & rename
        ds_ssrd = xr.open_dataset(ssrd_path) * 11.574
        ds_vpd = xr.open_dataset(vpd_path).rename({"vpd_cf": "vpd"})
        ds = xr.merge([ds_ssrd, ds_vpd])

        # ---- Compute GPP and SIF, plus uncertainties
        ds = SIF_GPP_EVI_model_ssrd(ds,use_f_ssrd)
        ds = SIF_GPP_EVI_model_ssrd_uncertainty(ds)
        ds['evi'] =  (ds['evi'] - EVI_mean) / EVI_std

        ds = compute_seasonal_pft_evi_forward(ds)

        # ---- Standardize the input and the target
        ds["ssrd"] = (ds["ssrd"] - SW_IN_mean) / SW_IN_std
        ds['gpp']  = (ds['gpp']  - GPP_mean)   / GPP_std
        # if pft_code:
        #     ds['is_tropical'] = xr.where(ds['pft'] == PFT['tropical'], 1.0, 0.0)
        #     ds['is_temperate'] = xr.where(ds['pft'] == PFT['temperate'], 1.0, 0.0)
        #     ds['is_boreal'] = xr.where(ds['pft'] == PFT['boreal'], 1.0, 0.0)
        #subsample the data
        ds_remaining = ds.where(mask_da)

        # ---- Flatten data for PyTorch
        # We'll retrieve not just X_data, y_data but also the lat/lon indices.
        X_data, y_data, lat_idx_arr, lon_idx_arr, time_idx_arr = preprocess_data_lat_lon(
            ds_remaining, predictors, target_gpp
        )

        batch_size = 2**8
        dataset_climate = ClimateDataset_lat_lon(X_data, y_data, lat_idx_arr, lon_idx_arr, time_idx_arr)
        dataloader = DataLoader(dataset_climate, batch_size=batch_size, shuffle=False)

        pred_gpp_array = np.full((ds.dims["time"], ds.dims["latitude"], ds.dims["longitude"]), np.nan)

        with torch.no_grad():
            for batch_X, batch_y, lat_batch, lon_batch, time_batch in dataloader:
                outputs = model(batch_X.float().to(device))
                # Bring predictions and targets back to CPU
                outputs_np = outputs.cpu().numpy().squeeze()
                batch_y_np = batch_y.numpy()
                # Invert the standardization if you want metrics in original GPP units
                outputs_np_orig = outputs_np * GPP_std + GPP_mean
                batch_y_np_orig = batch_y_np * GPP_std + GPP_mean

                # Valid (non-NaN) mask
                valid_mask = ~np.isnan(outputs_np_orig) & ~np.isnan(batch_y_np_orig)
                outputs_valid = outputs_np_orig[valid_mask]
                lat_idx_valid = lat_batch[valid_mask]
                lon_idx_valid = lon_batch[valid_mask]
                time_idx_valid = time_batch[valid_mask]

                # Ensure lat_idx_valid and lon_idx_valid are numpy arrays
                if torch.is_tensor(lat_idx_valid):
                    lat_idx_valid = lat_idx_valid.cpu().numpy()
                if torch.is_tensor(lon_idx_valid):
                    lon_idx_valid = lon_idx_valid.cpu().numpy()

                lat_indices = np.array([np.where(lat_vals == val)[0][0] for val in lat_idx_valid])
                lon_indices = np.array([np.where(lon_vals == val)[0][0] for val in lon_idx_valid])

                # Assign predictions to the correct positions in pred_gpp_array
                pred_gpp_array[time_idx_valid, lat_indices, lon_indices] = outputs_valid

        ds_remaining['gpp_hat'] = (("time","latitude","longitude"),pred_gpp_array)
        if pft_code:
            ds_remaining.to_netcdf("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/fine_tuning_forward_output_"+str(year_idx)+"_"+case+"_subsample.nc")
        else:
            ds_remaining.to_netcdf("/Net/Groups/BGI/people/xyu/sifluxcom/outputs/fine_tuning_forward_output_"+str(year_idx)+"_"+case+"_subsample.nc")
        # Clear cache
        torch.cuda.empty_cache()
        print(f"Processed year file: {ssrd_path.name}")
        print_memory_usage()