import os
import sys

sys.path.insert(0, "../")

import torch

torch.set_default_dtype(torch.float32)

import numpy as np
import h5py
from scipy.stats import binned_statistic
from unyt import unyt_quantity, Msun
from synthesizer.conversions import lnu_to_absolute_mag, absolute_mag_to_lnu
from camels import camels
import matplotlib.pyplot as plt

# Alice Added:

from variables_config import get_config
config = get_config()

def get_available_snapshots(photo_dir="/disk/xray15/aem2/data/6pams/"):
    """Get list of available snapshots from the HDF5 file."""
    available_snaps = set()
    with h5py.File(f"{photo_dir}/alice_galex_LH.h5", "r") as hf:
        # Get first simulation to check available snaps
        first_sim = list(hf.keys())[0]
        available_snaps = {k.split('_')[1] for k in hf[first_sim].keys() if k.startswith('snap_')}
    return sorted(list(available_snaps))

def get_photometry(
    sim_name="LH_0",
    spec_type="attenuated",
    snap="090",
    # snap=None,  # Made this None by default
    sps="BC03",
    model="IllustrisTNG",
    photo_dir=(
        "/disk/xray15/aem2/data/6pams/"
    ),
    filters=[
        "GALEX FUV",
        "GALEX NUV",
    ],
):
    # Rest of function stays the same
    photo_file = f"{photo_dir}/alice_galex_LH.h5"
    photo = {}
    with h5py.File(photo_file, "r") as hf:
        for filt in filters:
            # Updated path to match downloaded alice_galex_LH file structure
            photo[filt] = hf[
                f"{sim_name}/snap_{snap}/{sps}/photometry/luminosity/{spec_type}/{filt}"
            ][:]
            photo[filt] *= unyt_quantity.from_string("1 erg/s/Hz")
            photo[filt] = lnu_to_absolute_mag(photo[filt])

    return photo


def get_x(
    spec_type="attenuated",
    snap="090",
    sps="BC03",
    luminosity_functions=True,
    colours=True,
    model="IllustrisTNG",
    photo_dir=(
        "/disk/xray15/aem2/data/6pams/"
    ),
    n_bins_lf=13,
    n_bins_colour=13,
):
    if isinstance(snap, str):
        snap = [snap]

    x = [[] for _ in range(1000)]

    for _LH in np.arange(1000):
        for snp in snap:
            photo = get_photometry(
                sim_name=f"LH_{_LH}",
                spec_type=spec_type,
                snap=snp,
                sps=sps,
                model=model,
                photo_dir=photo_dir,
            )

            if luminosity_functions:
                for filt, lo_lim, hi_lim in zip(
                    [
                        "GALEX FUV",
                        "GALEX NUV",
                    ],
                    # lo_lim, hi_lim = config['uvlf_limits']  # Will be (-25, -14)
                    # why has chris defined them like this? run to test
                    [-21.5, -22.5, -23.5, -24, -24.5, -20.5, -20.5],
                    [-16, -17, -18, -18.5, -19, -15, -15],
                ):
                    phi = get_luminosity_function(
                        photo, filt, lo_lim, hi_lim, n_bins=n_bins_lf
                    )[0]
                    x[_LH].append(phi)

            if colours:
                for idx_A, idx_B, lo_lim, hi_lim in zip(
                    [
                        "GALEX FUV",
                    ],
                    [
                        "GALEX NUV",
                    ],
                    [0.0, 0.0, -0.1, -0.5],
                    [1.0, 0.5, 0.4, 3.5],
                ):
                    binLimsColour = np.linspace(lo_lim, hi_lim, n_bins_colour)

                    color = photo[idx_A] - photo[idx_B]
                    color_dist = np.histogram(
                        color, binLimsColour, density=True
                    )[0]

                    x[_LH].append(color_dist)

    return x

#############
# Plotting
#############
# Basic UVLF plot
def plot_uvlf(x_array, n_bins=13):
    """Plot UVLF for a specific simulation"""
    # Create magnitude bins
    mag_bins = np.linspace(-26, -14, n_bins)
    mag_centers = (mag_bins[:-1] + mag_bins[1:]) / 2
    
    # Calculate number of bins for LF data
    n_lf_bins = n_bins - 1
    
    # Extract just the LF part of the data
    fuv_bins = x_array[0, :n_lf_bins]
    nuv_bins = x_array[0, n_lf_bins:2*n_lf_bins]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(mag_centers, fuv_bins, 'o-', label='FUV', color='blue', alpha=0.7)
    ax.plot(mag_centers, nuv_bins, 's-', label='NUV', color='red', alpha=0.7)
    
    ax.set_xlabel('Magnitude (AB)', fontsize=12)
    ax.set_ylabel('log$_{10}$ φ [Mpc$^{-3}$ mag$^{-1}$]', fontsize=12)
    ax.legend()
    ax.grid(True)
    
    return fig
    
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


# Original from Chris Script:
def calc_df(_x, volume, massBinLimits):
    hist, _dummy = np.histogram(_x, bins=massBinLimits)
    hist = np.float64(hist)
    phi = (hist / volume) / (massBinLimits[1] - massBinLimits[0])

    phi_sigma = (np.sqrt(hist) / volume) / (
        massBinLimits[1] - massBinLimits[0]
    )  # Poisson errors

    return phi, phi_sigma, hist


def get_theta(
    model="IllustrisTNG",
    device="cuda",
):
    # dat = pd.read_csv('../data/dust_parameters.txt', delim_whitespace=True)
    cam = camels(model=model)

    theta = np.array(
        [
            cam.params['Omega_m'],
            cam.params['sigma_8'],
            cam.params['A_SN1'],
            cam.params['A_AGN1'],
            cam.params['A_SN2'],
            cam.params['A_AGN2'],
            # dat.tau_ism,
            # dat.tau_bc,
            # dat.UV_slope,
            # dat.OPT_NIR_slope,
            # dat.FUV_slope,
            # dat.bump,
        ]
    ).T

    return torch.tensor(theta, dtype=torch.float32, device=device)


def get_photometry_chris(
    sim_name="LH_0",
    spec_type="attenuated",
    snap="090",
    sps="BC03",
    model="IllustrisTNG",
    photo_dir=(
        "/mnt/home/clovell/code/" "camels_observational_catalogues/data/"
    ),
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


def get_mass_to_light(
    photo,
    solar_mag,
    halo_mass,
    filt,
    n_bins=15,
    mask=None,
):
    """
    Calculate binned mass to light ratio as a function of halo mass.
    Normalised by the solar (absolute) magnitude in the given filter.

    Args:
        photo (dict): Dictionary of photometry data.
        solar_mag (float): Solar (absolute) magnitude in the given filter.
        halo_mass (np.array): Halo masses.
        filt (str): Filter to calculate mass to light ratio in.
        n_bins (int): Number of bins to use.
        mask (np.array): Mask to apply to the data.

    Returns:
        ml_ratio (np.array): Binned mass to light ratio.
        binLimits (np.array): Bin limits.
        out_sources (np.array): Mass to light ratio of unbinned sources
    """
    if mask is None:
        mask = np.ones(len(photo[filt]))

    mag_ratio = 2.512 ** (solar_mag - photo[filt])

    mass_mask = halo_mass > 0.0
    mag_ratio = mag_ratio[mass_mask]
    halo_mass = halo_mass[mass_mask]

    binLimits = np.linspace(10, 16, n_bins)
    ml_ratio, _, _ = binned_statistic(
        np.log10(halo_mass),
        halo_mass / mag_ratio,
        statistic="median",
        bins=binLimits,
    )
    count, _, _ = binned_statistic(
        np.log10(halo_mass),
        halo_mass / mag_ratio,
        statistic="count",
        bins=binLimits,
    )

    mask = count >= 10
    halo_mass_lim = binLimits[1:][mask].max()
    out_sources = (halo_mass / mag_ratio)[np.log10(halo_mass) >= halo_mass_lim]

    return (
        ml_ratio,
        binLimits,
        mask,
        halo_mass_lim,
        (
            out_sources,
            halo_mass[np.log10(halo_mass) >= halo_mass_lim],
        ),
    )


def get_x_chris(
    spec_type="attenuated",
    snap="090",
    sps="BC03",
    luminosity_functions=True,
    colours=True,
    model="IllustrisTNG",
    photo_dir=(
        "/mnt/home/clovell/code/" "camels_observational_catalogues/data/"
    ),
    n_bins_lf=13,
    n_bins_colour=13,
):
    if isinstance(snap, str):
        snap = [snap]

    x = [[] for _ in range(1000)]

    for _LH in np.arange(1000):
        for snp in snap:
            photo = get_photometry(
                sim_name=f"LH_{_LH}",
                spec_type=spec_type,
                snap=snp,
                sps=sps,
                model=model,
                photo_dir=photo_dir,
            )

            if luminosity_functions:
                for filt, lo_lim, hi_lim in zip(
                    [
                        "SLOAN/SDSS.u",
                        "SLOAN/SDSS.g",
                        "SLOAN/SDSS.r",
                        "SLOAN/SDSS.i",
                        "SLOAN/SDSS.z",
                        "GALEX FUV",
                        "GALEX NUV",
                    ],
                    [-21.5, -22.5, -23.5, -24, -24.5, -20.5, -20.5],
                    [-16, -17, -18, -18.5, -19, -15, -15],
                ):
                    phi = get_luminosity_function(
                        photo, filt, lo_lim, hi_lim, n_bins=n_bins_lf
                    )[0]
                    x[_LH].append(phi)

            if colours:
                for idx_A, idx_B, lo_lim, hi_lim in zip(
                    [
                        "SLOAN/SDSS.g",
                        "SLOAN/SDSS.r",
                        "SLOAN/SDSS.i",
                        "GALEX FUV",
                    ],
                    [
                        "SLOAN/SDSS.r",
                        "SLOAN/SDSS.i",
                        "SLOAN/SDSS.z",
                        "GALEX NUV",
                    ],
                    [0.0, 0.0, -0.1, -0.5],
                    [1.0, 0.5, 0.4, 3.5],
                ):
                    binLimsColour = np.linspace(lo_lim, hi_lim, n_bins_colour)

                    color = photo[idx_A] - photo[idx_B]
                    color_dist = np.histogram(
                        color, binLimsColour, density=True
                    )[0]

                    x[_LH].append(color_dist)

    return x


def get_theta_x(
    spec_type="attenuated",
    snap="090",
    sps="BC03",
    model="IllustrisTNG",
    device="cuda",
    **kwargs,
):
    x = get_x(spec_type=spec_type, snap=snap, sps=sps, model=model, **kwargs)
    theta = get_theta(model=model, device=device)
    return theta, x


def get_intrinsic_properties(
    directory="/mnt/ceph/users/camels/FOF_Subfind/",
    photo_dir="/mnt/home/clovell/ceph/CAMELS_photometry/",
    model="IllustrisTNG",
    sim_set="1P",
    sim="1P_p1_0",
    snap="086",
    box="L25n256",
    verbose=False,
):
    if model == "Simba":
        _model = "SIMBA"
    else:
        _model = model

    # if _model == "Swift-EAGLE":
    # _f = f"/{directory}/{sim}/fof_subhalo_tab_{snap}.hdf5"
    # else:
    _f = f"/{directory}/{_model}/{box}/{sim_set}/{sim}/groups_{snap}.hdf5"

    if not os.path.isfile(_f):
        print("File %s not found, skipping" % _f)
    else:
        if verbose:
            print(f"Processing {_f}")

    with h5py.File(_f, "r") as hf:
        # subhalo_mass = hf["Group/Group_M_Crit200"][:] * 1e10
        subhalo_mass = hf["Subhalo/SubhaloMass"][:] * 1e10
        stellar_mass = hf["Subhalo/SubhaloMassType"][:][:, 4] * 1e10
        sfr = hf["Subhalo/SubhaloSFR"][:]
        bh_mass = hf["Subhalo/SubhaloBHMass"][:] * 1e10
        gas_mass = hf["Subhalo/SubhaloMassType"][:][:, 0] * 1e10

    photo_file = f"{photo_dir}/{model}/{model}_{sim}_photometry.hdf5"
    with h5py.File(photo_file, "r") as hf:
        subhalo_index = np.array(hf[f"snap_{snap}/SubhaloIndex"][:], dtype=int)

    return (
        subhalo_mass[subhalo_index],
        stellar_mass[subhalo_index],
        sfr[subhalo_index],
        bh_mass[subhalo_index],
        gas_mass[subhalo_index],
    )


if __name__ == "__main__":
    theta, x = get_theta_x()
    print(theta.shape, x.shape)
