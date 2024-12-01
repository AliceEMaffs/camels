import sys
import numpy as np
import pandas as pd
import h5py
from unyt import unyt_quantity
from synthesizer.conversions import lnu_to_absolute_mag
from camels import camels
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import os

sys.path.insert(0, "../")
torch.set_default_dtype(torch.float32)

def calc_df(_x, volume, massBinLimits):
    hist, _dummy = np.histogram(_x, bins=massBinLimits)
    hist = np.float64(hist)
    phi = (hist / volume) / (massBinLimits[1] - massBinLimits[0])

    phi_sigma = (np.sqrt(hist) / volume) / (
        massBinLimits[1] - massBinLimits[0]
    )  # Poisson errors

    return phi, phi_sigma, hist


def get_photometry(
    sim_name="LH_0",
    spec_type="attenuated",
    snap="090",
    sps="BC03",
    model="IllustrisTNG",
    photo_dir=("/disk/xray15/aem2/data/28pams/IllustrisTNG/photometry"),
    filters=[
        "SLOAN/SDSS.u",
        "SLOAN/SDSS.g",
        "SLOAN/SDSS.r",
        "SLOAN/SDSS.i",
        "SLOAN/SDSS.z",
        "GALEX FUV",
        "GALEX NUV",
    ],
):
    photo_file = f"{photo_dir}/{model}_{sim_name}_photometry.hdf5"
    photo = {}
    with h5py.File(photo_file, "r") as hf:
        for filt in filters:
            photo[filt] = hf[
                f"snap_{snap}/{sps}/photometry/luminosity/{spec_type}/{filt}"
            ][:]
            photo[filt] *= unyt_quantity.from_string("1 erg/s/Hz") 
            photo[filt] = lnu_to_absolute_mag(photo[filt])

    return photo
    

def get_luminosity_function(
    photo,
    filt,
    lo_lim,
    hi_lim,
    n_bins=15,
    mask=None,
):
    h = 0.6711
    if mask is None:
        mask = np.ones(len(photo[filt]), dtype=bool)

    binLimits = np.linspace(lo_lim, hi_lim, n_bins)
    phi, phi_sigma, hist = calc_df(photo[filt][mask], (25 / h) ** 3, binLimits)
    phi[phi == 0.0] = 1e-6 + np.random.rand() * 1e-7
    phi = np.log10(phi)
    return phi, phi_sigma, hist, binLimits


def get_colour_distribution(
    photo,
    filtA,
    filtB,
    lo_lim,
    hi_lim,
    n_bins=10,
    mask=None,
):
    if mask is None:
        mask = np.ones(len(photo[filtA]))

    binLimsColour = np.linspace(lo_lim, hi_lim, n_bins)
    color = (photo[filtA] - photo[filtB])[mask]
    colour_dist = np.histogram(color, binLimsColour, density=True)[0]
    return colour_dist, binLimsColour


#############
# processing data to Txts
#############

# safe name for passing in and using for naming paths/directories without underscores.  see more below (get_colour_dir_name)
def get_safe_name(name, filter_system_only=False):
    """
    Convert string to path-safe version and/or extract filter system.
    
    Args:
        name: String to process (e.g., "GALEX FUV" or "UV1500")
        filter_system_only: If True, returns only the filter system (e.g., "GALEX" or "UV")
    
    Returns:
        Processed string (e.g., "GALEX_FUV" or "GALEX")
    """
    # Replace spaces with underscores
    safe_name = name.replace(' ', '_')
    
    # If we only want the filter system, return the first part
    if filter_system_only:
        return safe_name.split('_')[0]
    
    return safe_name


def get_colour_dir_name(band1, band2):
    """
    Create a standardized directory name for colour plots.
    Examples:
        ("GALEX FUV", "GALEX NUV") -> "GALEX_FUV-NUV"
        ("UVM2", "SUSS") -> "UVM2-SUSS"
    """
    # Extract the relevant parts of the filter names
    if ' ' in band1:
        system1, filter1 = band1.split(' ', 1)
    else:
        system1, filter1 = band1, band1

    if ' ' in band2:
        system2, filter2 = band2.split(' ', 1)
    else:
        system2, filter2 = band2, band2
        
    # If both filters are from the same system, use shortened version
    if system1 == system2:
        return f"{get_safe_name(system1)}_{filter1}-{filter2}"
    else:
        return f"{get_safe_name(band1)}-{get_safe_name(band2)}"
    

def get_magnitude_mask(photo, filters, mag_limits=None):
    """
    Create a magnitude mask based on provided limits.
    
    Args:
        photo (dict): Photometry data dictionary
        filters (list): List of filters to check
        mag_limits (dict): Dictionary of magnitude limits for each filter
    
    Returns:
        numpy.ndarray: Boolean mask array, or None if no limits provided
    """
    if not mag_limits:
        return None
        
    # Start with all True
    combined_mask = np.ones(len(photo[filters[0]]), dtype=bool)
    
    # Apply limits for each filter
    for band in filters:
        if band in mag_limits:
            combined_mask &= (photo[band] < mag_limits[band])
            
    return combined_mask

def process_data(input_dir, redshift_values, uvlf_limits, n_bins_lf, lf_data_dir, colour_limits, n_bins_colour, colour_data_dir, category, bands, colour_pairs=None, mag_limits=None):
    """Process data for any combination of bands and color pairs"""
    photo_files = [f for f in os.listdir(input_dir) if f.endswith('_photometry.hdf5')]
    
    for filename in photo_files:
        sim_name = filename.replace('IllustrisTNG_', '').replace('_photometry.hdf5', '')
        print(f"\nProcessing {sim_name}")
        
        for snap, redshift_info in redshift_values.items():
            print(f"  Processing z={redshift_info['label']}")
            
            # Get photometry
            photo = get_photometry(
                sim_name=sim_name,
                spec_type="intrinsic" if category == "intrinsic" else "attenuated",
                snap=snap,
                sps="BC03",
                model="IllustrisTNG",
                filters=bands,
                photo_dir=input_dir
            )
            
            # Process UVLFs
            for band in bands:
                phi, phi_sigma, hist, bin_lims = get_luminosity_function(
                    photo,
                    band,
                    *uvlf_limits,
                    n_bins=n_bins_lf
                )
                
                # Save UVLF
                bin_centers = 0.5 * (bin_lims[1:] + bin_lims[:-1])
                uvlf_df = pd.DataFrame({
                    'magnitude': bin_centers,
                    'phi': phi,
                    'phi_sigma': phi_sigma,
                    'hist': hist
                })
                
                # Get filter system and create output directory
                filter_system = get_safe_name(band, filter_system_only=True)
                spec_type = "intrinsic" if category == "intrinsic" else "attenuated"
                
                output_dir = os.path.join(lf_data_dir[category][filter_system], get_safe_name(redshift_info['label']))
                os.makedirs(output_dir, exist_ok=True)
                
                # Use get_safe_name for the band in the filename
                uvlf_filename = f"UVLF_{sim_name}_{get_safe_name(band)}_{get_safe_name(redshift_info['label'])}_{spec_type}.txt"
                uvlf_df.to_csv(os.path.join(output_dir, uvlf_filename), 
                             index=False, sep='\t')
            
            # Process colours if pairs provided
            if colour_pairs:
                for band1, band2 in colour_pairs:
                    if band1 in photo and band2 in photo:
                        # Get mask using magnitude limits
                        mask = get_magnitude_mask(photo, [band1, band2], mag_limits)
                        
                        # Calculate color distribution using original function
                        colour_dist, bin_lims = get_colour_distribution(
                            photo,
                            band1,
                            band2,
                            *colour_limits,  # Unpack the min/max limits from config
                            n_bins=n_bins_colour,
                            mask=mask
                        )
                        
                        bin_centers = 0.5 * (bin_lims[1:] + bin_lims[:-1])
                        colour_df = pd.DataFrame({
                            'colour': bin_centers,
                            'distribution': colour_dist
                        })

                        # Get filter system and create output directory
                        filter_system = get_colour_dir_name(band1, band2)
                        output_dir = os.path.join(colour_data_dir[category][filter_system], get_safe_name(redshift_info['label']))
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # Create standardized filename using get_safe_name for band names
                        colour_filename = f"Colour_{sim_name}_{filter_system}_{get_safe_name(redshift_info['label'])}_{spec_type}.txt"
                        colour_df.to_csv(os.path.join(output_dir, colour_filename),
                                       index=False, sep='\t')

            print(f"    Completed processing for z={redshift_info['label']}")

