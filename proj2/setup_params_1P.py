import sys
import numpy as np
import pandas as pd
import h5py
from unyt import unyt_quantity
from synthesizer.conversions import lnu_to_absolute_mag
from camels_SB import camels
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

def process_data(input_dir, redshift_values, uvlf_limits, uvlf_nbins, lf_data_dir, colour_limits, colour_nbins, colour_data_dir, category, bands, colour_pairs=None, mag_limits=None):
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
                    n_bins=uvlf_nbins
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
                            n_bins=colour_nbins,
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

#############
# plotting
#############
# basic UVLF plot
def plot_uvlf(x_array, n_bins=13):
    """Plot UVLF for a specific simulation"""
    # Create magnitude bins
    mag_bins = np.linspace(-27, -17, n_bins)
    mag_centers = (mag_bins[:-1] + mag_bins[1:]) / 2
    
    # Calculate number of bins for LF data
    n_lf_bins = n_bins - 1
    
    # Extract just the LF part of the data
    fuv_bins = x_array[0, :n_lf_bins]
    nuv_bins = x_array[0, n_lf_bins:2*n_lf_bins]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(mag_centers, fuv_bins, 'o-', label='FUV', color='blue', alpha=0.7)
    ax.plot(mag_centers, nuv_bins, 's-', label='NUV', color='red', alpha=0.7)
    
    ax.set_xlabel('M$_{UV}$')
    ax.set_ylabel('log$_{10}$ φ [Mpc$^{-3}$ mag$^{-1}$]')
    ax.legend()
    ax.grid(True)
    
    return fig
    
import numpy as np
import matplotlib.pyplot as plt

def plot_colour(x_array, n_bins=13, n_sims_to_plot=5): # taking a sample of 5 and taking the median
    """
    Plot FUV-NUV color distributions for multiple simulations.
    
    Parameters:
    -----------
    x_array : numpy.ndarray
        Array containing the color distributions (last n_bins elements of each row)
    n_bins : int
        Number of bins used for the color histogram
    n_sims_to_plot : int
        Number of random simulations to plot (to avoid overcrowding)
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Create color bins
    color_bins = np.linspace(-0.5, 3.5, n_bins)
    color_centers = (color_bins[:-1] + color_bins[1:]) / 2
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot individual simulation distributions
    random_indices = np.random.choice(len(x_array), size=n_sims_to_plot, replace=False)
    for idx in random_indices:
        # Get color distribution (last n_bins-1 elements)
        color_dist = x_array[idx, -n_bins+1:]
        ax.plot(color_centers, color_dist, alpha=0.3, color='gray', linewidth=1)
    
    # Plot mean distribution
    mean_dist = np.mean(x_array[:, -n_bins+1:], axis=0)
    ax.plot(color_centers, mean_dist, 'b-', linewidth=2, label='Mean Distribution')
    
    # Add labels and styling
    ax.set_xlabel('FUV - NUV [mag]')
    ax.set_ylabel('Normalized Count')
    ax.set_title('FUV-NUV Color Distribution')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Print ranges for verification
    print(f"Color range: [{color_bins[0]:.1f}, {color_bins[-1]:.1f}]")
    print(f"Distribution range: [{mean_dist.min():.2f}, {mean_dist.max():.2f}]")
    
    return fig


def calculate_parameter_values(param_info):
    """
    Calculate all parameter values based on LogFlag.
    Returns dictionary with values for each variation (n2, n1, 0, 1, 2).
    """
    min_val = param_info['MinVal']
    max_val = param_info['MaxVal']
    fid_val = param_info['FiducialVal']
    
    if param_info['LogFlag']:
        # Logarithmic spacing
        log_min = np.log10(min_val)
        log_max = np.log10(max_val)
        log_fid = np.log10(fid_val)
        
        # Calculate n1 and 1 values in log space
        n1_val = 10**(log_min + (log_fid - log_min)/2)
        val_1 = 10**(log_fid + (log_max - log_fid)/2)
    else:
        # Linear spacing
        n1_val = min_val + (fid_val - min_val)/2
        val_1 = fid_val + (max_val - fid_val)/2
    
    return {
        'n2': min_val,    # MinVal
        'n1': n1_val,     # Calculated intermediate value
        '0': fid_val,     # FiducialVal
        '1': val_1,       # Calculated intermediate value
        '2': max_val      # MaxVal
    }

def read_colour_file(filename):
    """Read data from file."""
    try:
        data = pd.read_csv(filename, delim_whitespace=True)
        return data
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None
    
def plot_parameter_variations_uvlf(param_num, param_info, redshift, band_type, band, lf_data_dir, uvlf_limits, plots_dir_1P):
    """Plot UVLF showing all parameter variations at one redshift."""
    # This is your existing plot_all_uvlf_variations function
    param_values = calculate_parameter_values(param_info)
    
    plt.figure(figsize=(10, 8))
    variations = ['n2', 'n1', '1', '2']
    colours = ['blue', 'green', 'red', 'purple']
    
    # Get filter system from band name
    filter_system = get_safe_name(band, filter_system_only=True)
    base_output_dir = lf_data_dir[band_type][filter_system]
    
    # Plot each variation
    for var, colour in zip(variations, colours):
        filename = os.path.join(get_safe_name(base_output_dir), redshift['label'],
                              f"UVLF_1P_p{param_num}_{var}_{get_safe_name(band)}_{redshift['label']}_{band_type}.txt")
        
        if os.path.exists(filename):
            data = pd.read_csv(filename, delimiter='\t')
            label = f"{param_info['ParamName']} = {param_values[var]:.3g}"
            plt.errorbar(data['magnitude'], data['phi'], 
                        yerr=data['phi_sigma'],
                        fmt='o-', color=colour, 
                        label=label,
                        markersize=4, capsize=2)
        else:
            print(f"File not found: {filename}")

    if plt.gca().has_data():
        plt.xlabel('Magnitude (AB)', fontsize=12)
        plt.ylabel('Number Density (Mpc$^{-3}$ mag$^{-1}$)', fontsize=12)
        plt.xlim(*uvlf_limits)
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.title(f'{param_info["Description"]} - {param_info["ParamName"]} (p{param_num})\n'
                 f'z = {redshift["redshift"]}',
                 fontsize=14)
        plt.legend(loc='upper left', title='Parameter Values')
        plt.figtext(0.02, 0.02, f"LogFlag: {param_info['LogFlag']}", fontsize=8)
        
        # Updated path to match new structure
        output_path = os.path.join(plots_dir_1P, 'LFs', band_type, get_safe_name(band), 
                                 'parameter_variations',
                                 f'UVLF_p{param_num}_{param_info["ParamName"]}_z{redshift["label"]}.pdf')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Saving UVLF plot: {output_path}")
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    
    plt.close()

def plot_parameter_variations_colour(param_num, param_info, redshift, band_type, colour_pairs, colour_limits, colour_data_dir, plots_dir_1P):
    """Plot colour distribution showing all parameter variations at one redshift."""
    param_values = calculate_parameter_values(param_info)
    variations = ['n2', 'n1', '1', '2']
    colours = ['blue', 'green', 'red', 'purple']
    
    for band1, band2 in colour_pairs:
        plt.figure(figsize=(10, 8))
        
        # Use new function for directory name
        colour_dir = get_colour_dir_name(band1, band2)
        
        for var, colour in zip(variations, colours):
            # Create consistent filename
            colour_name = f"{band1.split()[-1]}-{band2.split()[-1]}"  # Keep original naming for files
            
            filename = os.path.join(colour_data_dir[band_type][colour_dir], 
                                  redshift['label'],
                                  f"Colour_1P_p{param_num}_{var}_{colour_name}_{redshift['label']}_{band_type}.txt")
            
            if os.path.exists(filename):
                data = pd.read_csv(filename, delimiter='\t')
                plt.plot(data['colour'], data['distribution'], 
                        color=colour, linewidth=2,
                        label=f'{param_info["ParamName"]} = {param_values[var]:.3g}')
        
        plt.xlabel(f'{band1} - {band2} [mag]', fontsize=14)
        plt.ylabel('Normalized Count', fontsize=14)
        plt.xlim(*colour_limits)
        plt.grid(True, which='both', linestyle='--', alpha=0.3)
        plt.title(f'{param_info["Description"]} - {param_info["ParamName"]} (p{param_num})\n'
                 f'z = {redshift["redshift"]}', 
                 fontsize=16)
        plt.legend(title='Parameter Values', 
                  title_fontsize=12,
                  fontsize=11,
                  loc='upper right',
                  framealpha=0.95)
        
        plt.figtext(0.02, 0.02, f"LogFlag: {param_info['LogFlag']}", fontsize=8)
        plt.tight_layout()
        
        # Updated output path using new directory name
        output_dir = os.path.join(plots_dir_1P, 'colours', band_type, colour_dir, 
                                'parameter_variations')
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 
                                  f'Colour_p{param_num}_{param_info["ParamName"]}_z{redshift["label"]}.pdf')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()



def process_all_parameters(param_info_file, redshift_values, filters):
    """Process all parameter variation plots."""
    colour_pairs = colour_pairs
    
    params_df = pd.read_csv(param_info_file)
    
    for param_num in range(1, 29):
        param_info = params_df.iloc[param_num-1]
        print(f"\nProcessing parameter {param_num}: {param_info['ParamName']}")
        
        for snap, redshift in redshift_values.items():
            print(f"Processing redshift {redshift['redshift']}")
            
            for band_type in ['intrinsic', 'attenuated']:
                # Process UVLFs
                for band in filters[band_type]:
                    print(f"Plotting UVLF variations for {band_type} {band}")
                    plot_parameter_variations_uvlf(param_num, param_info, redshift, band_type, band)
                
                # Process colours
                print(f"Processing colour variations for z={redshift['redshift']}")
                plot_parameter_variations_colour(param_num, param_info, redshift, band_type, colour_pairs)


def process_lfs_all(redshift_values, filters, param_info_file, lf_data_dir, plots_dir_1P,uvlf_limits):
    """Process UVLF parameter grids for all redshifts."""
    
    print("Available redshifts:", redshift_values.keys())  # Debug print
    
    for snap, redshift_info in redshift_values.items():
        print(f"\nProcessing redshift {redshift_info['redshift']} (snap {snap})")
        
        for band_type, band_list in filters.items():
            print(f"Processing band type: {band_type}")
            print(f"Bands to process: {band_list}")
            
            for band in band_list:
                print(f"Creating UVLF parameter grid for {band_type} {band}")
                create_lf_all(
                    param_info_file=param_info_file,
                    redshift=redshift_info,
                    band_type=band_type,
                    band_or_colour_pairs=band,
                    lf_data_dir=lf_data_dir,
                    plots_dir_1P=plots_dir_1P,
                    uvlf_limits=uvlf_limits
                )
                print(f"Completed processing {band} for z={redshift_info['redshift']}")

def process_colours_all(redshift_values, colour_pairs, param_info_file,colour_data_dir,colour_limits,plots_dir_1P):
    """Process color parameter grids for all redshifts.
    
    Args:
        redshift_values (dict): Dictionary of redshift information
        colour_pairs (list): List of color pairs to process
    """
    #params_df = pd.read_csv(param_info_file)
    
    for snap, redshift_info in redshift_values.items():
        print(f"\nProcessing redshift {redshift_info['redshift']}")
        
        for band_type in ['intrinsic', 'attenuated']:
            print(f"Creating colour parameter grid for {band_type}")
            create_colour_all(# params_df, redshift, band_type, band_or_colour_pairs, colour_data_dir, colour_limits, plots_dir_1P
                param_info_file=param_info_file,
                redshift=redshift_info,
                band_type=band_type,
                band_or_colour_pairs=colour_pairs,
                colour_data_dir=colour_data_dir,
                colour_limits=colour_limits, 
                plots_dir_1P=plots_dir_1P
                )

def plot_single_measurement(uvlf_file, color_file, band_info=None, color_info=None, output_dir=None):
    """
    Plot a single UVLF and color distribution from their respective txt files.
    
    Args:
        uvlf_file (str): Path to UVLF txt file
        color_file (str): Path to color distribution txt file
        band_info (tuple): (band_name, redshift, category) for plot labels
        color_info (tuple): (band1, band2, redshift, category) for plot labels
        output_dir (str): Directory to save plots. If None, just displays them.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot UVLF
    if os.path.exists(uvlf_file):
        uvlf_data = pd.read_csv(uvlf_file, delimiter='\t')
        ax1.errorbar(uvlf_data['magnitude'], uvlf_data['phi'], 
                    yerr=uvlf_data['phi_sigma'], 
                    fmt='o-', color='blue', capsize=5)
        
        ax1.set_xlabel('Absolute Magnitude (AB)', fontsize=12)
        ax1.set_ylabel('Number Density (Mpc$^{-3}$ mag$^{-1}$)', fontsize=12)
        if band_info:
            band, z, cat = band_info
            ax1.set_title(f'{band} UVLF\n(z={z}, {cat})', fontsize=14)
        ax1.grid(True, alpha=0.3)
    else:
        print(f"UVLF file not found: {uvlf_file}")
    
    # Plot Color Distribution
    if os.path.exists(color_file):
        color_data = pd.read_csv(color_file, delimiter='\t')
        ax2.plot(color_data['colour'], color_data['distribution'], 
                '-', color='blue', linewidth=2)
        ax2.fill_between(color_data['colour'], color_data['distribution'], 
                        alpha=0.3, color='blue')
        
        ax2.set_xlabel('Color [mag]', fontsize=12)
        ax2.set_ylabel('Normalized Count', fontsize=12)
        if color_info:
            band1, band2, z, cat = color_info
            ax2.set_title(f'{band1}-{band2} Color Distribution\n(z={z}, {cat})', fontsize=14)
        ax2.grid(True, alpha=0.3)
    else:
        print(f"Color file not found: {color_file}")
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'single_measurement.pdf'), 
                   bbox_inches='tight', dpi=300)
    else:
        plt.show()
    
    plt.close()


def create_redshift_evolution_plot(param_num, param_info, redshift_values, band_type, band, lf_data_dir, plots_dir_1P, uvlf_limits):
    """Create plot showing redshift evolution for each variation of a parameter."""
    param_values = calculate_parameter_values(param_info)
    
    variations = ['n2', 'n1', '1', '2']
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Redshift Evolution - {param_info["ParamName"]} (p{param_num})\n{param_info["Description"]}', 
                 fontsize=16, y=1.02)
    
    redshift_colours = {
        '044': 'darkred',     # z=2.00
        '052': 'orangered',   # z=1.48
        '060': 'orange',      # z=1.05
        '086': 'gold'         # z=0.10
    }
    
    filter_system = get_safe_name(band, filter_system_only=True)
    base_output_dir = lf_data_dir[band_type][filter_system]
    
    for ax, variation in zip(axes.flatten(), variations):
        for snap, redshift in redshift_values.items():
            filename = os.path.join(get_safe_name(base_output_dir), 
                                  redshift['label'],
                                  f"UVLF_1P_p{param_num}_{variation}_{get_safe_name(band)}_{redshift['label']}_{band_type}.txt")
            
            if os.path.exists(filename):
                try:
                    data = pd.read_csv(filename, delimiter='\t')
                    
                    ax.errorbar(data['magnitude'], data['phi'], 
                              yerr=data['phi_sigma'],
                              fmt='o-', color=redshift_colours[snap],
                              label=f'z = {redshift["redshift"]}',
                              markersize=4, capsize=2)
                except Exception as e:
                    print(f"Error plotting {filename}: {e}")
        
        ax.set_xlim(*uvlf_limits)
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        ax.set_title(f'{param_info["ParamName"]} = {param_values[variation]:.3g}', 
                    fontsize=12)
        ax.set_xlabel('Magnitude (AB)', fontsize=12)
        ax.set_ylabel('Φ (Mpc$^{-3}$ mag$^{-1}$)', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.legend(fontsize=10, title='Redshift', title_fontsize=10)
    
    plt.figtext(0.02, 0.02, f"LogFlag: {param_info['LogFlag']}", fontsize=8)
    plt.tight_layout()
    
    output_dir = os.path.join(plots_dir_1P, 'LFs', band_type, get_safe_name(band), 'redshift_variations')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 
                              f'UVLF_redshift_evolution_p{param_num}_{get_safe_name(param_info["ParamName"])}.pdf')
    print('Saving to', output_path)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def create_parameter_variation_plot(param_num, param_info, redshift_values, lf_data_dir, plots_dir_1P, band_type, band, uvlf_limits):
    """Create plot showing parameter variations at each redshift."""
    param_values = calculate_parameter_values(param_info)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Parameter Variations - {param_info["ParamName"]} (p{param_num})\n{param_info["Description"]}', 
                 fontsize=16, y=1.02)
    
    variation_colours = {
        'n2': 'blue',
        'n1': 'green',
        '1': 'red',
        '2': 'purple'
    }
    
    filter_system = get_safe_name(band, filter_system_only=True)
    base_output_dir = lf_data_dir[band_type][filter_system]
    
    for ax, (snap, redshift) in zip(axes.flatten(), redshift_values.items()):
        for var in ['n2', 'n1', '1', '2']:
            filename = os.path.join(get_safe_name(base_output_dir), 
                                  redshift['label'],
                                  f"UVLF_1P_p{param_num}_{var}_{get_safe_name(band)}_{redshift['label']}_{band_type}.txt")
            
            if os.path.exists(filename):
                try:
                    data = pd.read_csv(filename, delimiter='\t')
                    label = f'{param_info["ParamName"]} = {param_values[var]:.3g}'
                    
                    ax.errorbar(data['magnitude'], data['phi'], 
                              yerr=data['phi_sigma'],
                              fmt='o-', color=variation_colours[var],
                              label=label,
                              markersize=4, capsize=2)
                except Exception as e:
                    print(f"Error plotting {filename}: {e}")
        
        ax.set_xlim(*uvlf_limits)
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        ax.set_title(f'z = {redshift["redshift"]}', fontsize=12)
        ax.set_xlabel('Magnitude (AB)', fontsize=12)
        ax.set_ylabel('Φ (Mpc$^{-3}$ mag$^{-1}$)', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.legend(fontsize=9, title='Parameter Values', title_fontsize=10)
    
    plt.figtext(0.02, 0.02, f"LogFlag: {param_info['LogFlag']}", fontsize=8)
    plt.tight_layout()
    
    output_dir = os.path.join(plots_dir_1P, 'LFs', band_type, get_safe_name(band), 'parameter_variations')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 
                              f'UVLF_parameter_variation_p{param_num}_{get_safe_name(param_info["ParamName"])}.pdf')
    print('Saving to', output_path)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

# process redshift and parameter variation plots. (4 plots per fig)
def process_all_variations(param_info_file, filters, redshift_values, lf_data_dir, plots_dir_1P, uvlf_limits):
    """Process both types of variation plots for all parameters and bands.
    
    Args:
        param_info_file (str): Path to parameter info file
        filters (dict): Dictionary of filter configurations
        redshift_values (dict): Dictionary of redshift information
        lf_data_dir (dict): Directory for luminosity function data
        plots_dir_1P (str): Directory for saving plots
    """
    params_df = pd.read_csv(param_info_file)
    
    for param_num in range(1, 29):
        param_info = params_df.iloc[param_num-1]
        print(f"\nProcessing parameter {param_num}: {param_info['ParamName']}")
        
        for band_type in ['intrinsic', 'attenuated']:
            for band in filters[band_type]:
                print(f"Creating variation plots for {band_type} {band}")
                create_redshift_evolution_plot(param_num, param_info, redshift_values, band_type, band, lf_data_dir, plots_dir_1P, uvlf_limits)
                create_parameter_variation_plot(param_num, param_info, redshift_values, lf_data_dir, plots_dir_1P, band_type, band, uvlf_limits)

def create_colour_all(param_info_file, redshift, band_type, band_or_colour_pairs, colour_data_dir, colour_limits, plots_dir_1P):
    """
    Create grid plot of all parameters for a given redshift for colors.
    """
    ncols, nrows = 7, 4  # 28 parameters in a 7x4 grid
    fig, axes = plt.subplots(nrows, ncols, figsize=(35, 20))
    
    variations = ['n2', 'n1', '1', '2']
    colours = ['blue', 'green', 'red', 'purple']
    
    for param_num in range(1, 29):
        col = (param_num - 1) // nrows
        row = (param_num - 1) % nrows
        ax = axes[row, col]
        
        params_df = pd.read_csv(param_info_file)
        param_info = params_df.iloc[param_num-1]
        param_values = calculate_parameter_values(param_info)
        
        band1, band2 = band_or_colour_pairs[0]  # Assuming single colour pair
        filter_system = get_colour_dir_name(band1, band2)
        
        data_dir = os.path.join(colour_data_dir[band_type][filter_system], get_safe_name(redshift['label']))
        
        for var, colour in zip(variations, colours):
            filename = os.path.join(data_dir, f"Colour_1P_p{param_num}_{var}_{filter_system}_{get_safe_name(redshift['label'])}_{band_type}.txt")
            print(f"Checking color file: {filename}")
            
            if os.path.exists(filename):
                try:
                    data = pd.read_csv(filename, delimiter='\t')
                    print(f"Color data columns: {data.columns}")
                    ax.plot(data['colour'], data['distribution'], color=colour, linewidth=1.5,
                           label=f'{param_values[var]:.3g}')
                except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                    print(f"Error reading color file: {filename}")
                    print(e)
            else:
                print(f"Color file not found: {filename}")
                
    # Customize subplots
    for ax in axes.flatten():
        ax.set_xlim(*colour_limits)
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=6)
    
    # Add parameter labels and LogFlag info
    for param_num, ax in enumerate(axes.flatten(), start=1):
        param_info = params_df.iloc[param_num-1]
        ax.set_title(f'p{param_num}: {param_info["ParamName"]}', fontsize=8)
        ax.text(0.02, 0.98, f'LogFlag: {param_info["LogFlag"]}',
                transform=ax.transAxes, fontsize=6, va='top')
    
    # Add color bar
    cmap = mpl.colors.ListedColormap(colours)
    norm = mpl.colors.BoundaryNorm([-2.5, -1.5, -0.5, 0.5, 1.5], cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.375, 0.95, 0.25, 0.01])  # change position and size here
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', ticks=[-2, -1, 0, 1])
    cbar.ax.set_xticklabels(['Min', '', '0', 'Max'])
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Param values', fontsize=14)
    
    # Add overall title
    title = f'{filter_system} Colour Distribution {param_info["ParamName"]} Variations (z = {redshift["redshift"]})'
    fig.suptitle(title, fontsize=16, y=0.99)
    
    # Adjust subplot spacing to make room for color bar
    plt.subplots_adjust(top=0.93, bottom=0.05, hspace=0.3, wspace=0.2)
    
    # Save plot
    plot_dir = os.path.join(plots_dir_1P, 'colours', band_type, filter_system, get_safe_name(redshift['label']))
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f"All_colours_{filter_system}_{get_safe_name(redshift['label'])}_{band_type}.pdf"), bbox_inches='tight', dpi=300)
    plt.close()


def create_lf_all(param_info_file, redshift, band_type, band_or_colour_pairs, lf_data_dir, plots_dir_1P, uvlf_limits):
    """Create grid plot of all parameters for a given redshift for LFs."""
    print("\n=== DETAILED DEBUG INFO ===")
    print(f"Parameters received:")
    print(f"param_info_file: {param_info_file}")
    print(f"redshift: {redshift}")
    print(f"band_type: {band_type}")
    print(f"band_or_colour_pairs: {band_or_colour_pairs}")
    print(f"uvlf_limits: {uvlf_limits}")
    
    # Verify param_info_file exists
    if not os.path.exists(param_info_file):
        print(f"ERROR: param_info_file does not exist: {param_info_file}")
        return
        
    # Setup directories with verbose logging
    filter_system = get_safe_name(band_or_colour_pairs, filter_system_only=True)
    band_name = get_safe_name(band_or_colour_pairs)
    data_dir = os.path.join(lf_data_dir[band_type][filter_system], get_safe_name(redshift['label']))
    
    print(f"\nDirectory structure:")
    print(f"filter_system: {filter_system}")
    print(f"band_name: {band_name}")
    print(f"data_dir: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory does not exist: {data_dir}")
        return
    
    try:
        params_df = pd.read_csv(param_info_file)
        print(f"\nSuccessfully loaded parameter info file with {len(params_df)} parameters")
    except Exception as e:
        print(f"ERROR: Failed to read param_info_file: {str(e)}")
        return
    
    nrows, ncols = 7, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 35))
    axes = axes.flatten()
    
    variations = ['n2', 'n1', '1', '2']
    colours = ['blue', 'green', 'red', 'purple']
    plots_made = 0
    
    # Print first few expected filenames
    print("\nChecking for data files (first 3 parameters):")
    for param_num in range(1, 4):
        for var in variations:
            filename = os.path.join(data_dir, 
                                  f"UVLF_1P_p{param_num}_{var}_{band_name}_{get_safe_name(redshift['label'])}_{band_type}.txt")
            print(f"Looking for: {filename}")
            print(f"File exists: {os.path.exists(filename)}")
    
    for param_num in range(1, 29):
        ax = axes[param_num-1]
        param_info = params_df.iloc[param_num-1]
        param_values = calculate_parameter_values(param_info)
        
        panel_plots = 0
        print(f"\nProcessing parameter {param_num}: {param_info['ParamName']}")
        
        for var, colour in zip(variations, colours):
            filename = os.path.join(data_dir, 
                                  f"UVLF_1P_p{param_num}_{var}_{band_name}_{get_safe_name(redshift['label'])}_{band_type}.txt")
            
            if os.path.exists(filename):
                try:
                    data = pd.read_csv(filename, delimiter='\t')
                    print(f"  Variation {var}:")
                    print(f"    Rows in data: {len(data)}")
                    print(f"    Magnitude range: [{data['magnitude'].min():.2f}, {data['magnitude'].max():.2f}]")
                    print(f"    Phi range: [{data['phi'].min():.2e}, {data['phi'].max():.2e}]")
                    
                    ax.plot(data['magnitude'], data['phi'], 
                          color=colour, linewidth=1.5,
                          label=f'{param_values[var]:.3g}')
                    panel_plots += 1
                    plots_made += 1
                except Exception as e:
                    print(f"ERROR reading/plotting {filename}: {str(e)}")
            else:
                print(f"  File not found: {os.path.basename(filename)}")
        
        if panel_plots > 0:
            ax.set_xlim(*uvlf_limits)
            ax.set_ylim(-6, -2)
            ax.grid(True, which='both', linestyle='--', alpha=0.3)
            ax.set_title(f'p{param_num}: {param_info["ParamName"]}', fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=6)
            
            if param_num == 1:
                ax.legend(fontsize=6, title='Values', title_fontsize=6)
    
    print(f"\nFinal summary:")
    print(f"Total plots made: {plots_made}")
    
    if plots_made == 0:
        print("WARNING: No data was plotted! Check the debug information above for details.")
        plt.close()
        return
    
    # Save plot
    plot_dir = os.path.join(plots_dir_1P, 'LFs', band_type, band_name, get_safe_name(redshift['label']))
    os.makedirs(plot_dir, exist_ok=True)
    
    output_file = os.path.join(plot_dir, 
                              f"All_UVLF_{band_name}_{get_safe_name(redshift['label'])}_{band_type}.pdf")
    print(f"\nSaving plot to: {output_file}")
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
