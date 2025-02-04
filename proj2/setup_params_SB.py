import sys
import numpy as np
import pandas as pd
import h5py
from unyt import unyt_quantity
from synthesizer.conversions import lnu_to_absolute_mag
from camels_SB import camels
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
    photo_dir=("/disk/xray15/aem2/data/28pams/IllustrisTNG/SB/photometry"),
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
# SB STUFF
#############
# mask for test fraction (10% currently)
def create_test_mask(n_sims=2048, test_fraction=0.1, random_seed=42):
    """
    Create a test mask similar to the LH one but for SB 2048 simulations.
    
    Args:
        n_sims: Number of simulations (2048 for SB28)
        test_fraction: Fraction of simulations to use for testing (0.1 = 10%)
        random_seed: Random seed for reproducibility
    
    Returns:
        np.array: Boolean mask where True indicates test set
    """
    np.random.seed(random_seed)
    test_mask = np.random.rand(n_sims) > (1 - test_fraction)
    
    # Print some statistics
    print(f"Total simulations: {n_sims}")
    print(f"Test set size: {test_mask.sum()}")
    print(f"Training set size: {(~test_mask).sum()}")
    print(f"Test fraction: {test_mask.sum() / n_sims:.3f}")
    
    # Save the mask
    save_path = "/disk/xray15/aem2/data/28pams/IllustrisTNG/SB/test/test_mask_SB28.txt"
    np.savetxt(save_path, test_mask.astype(int), fmt='%i')
    
    return test_mask

'''  reference from setup_params_alice.py
def calc_df(_x, volume, massBinLimits):
    hist, _dummy = np.histogram(_x, bins=massBinLimits)
    hist = np.float64(hist)
    phi = (hist / volume) / (massBinLimits[1] - massBinLimits[0])

    phi_sigma = (np.sqrt(hist) / volume) / (
        massBinLimits[1] - massBinLimits[0]
    )  # Poisson errors

    return phi, phi_sigma, hist

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
'''
def get_photometry_SB(
    sim_name="SB28_0",
    spec_type="attenuated",
    snap="044",
    sps="BC03",
    model="IllustrisTNG",
    photo_dir="/disk/xray15/aem2/data/28pams/IllustrisTNG/SB/photometry",
    filters=["GALEX FUV", "GALEX NUV"],
):
    photo = {}
    with h5py.File(f"{photo_dir}/alice_galex.h5", "r") as hf:
        for filt in filters:
            path = f"{sim_name}/snap_{snap}/{sps}/photometry/luminosity/{spec_type}/{filt}"
            photo[filt] = hf[path][:]
            # comment out if data is already in desired format, remove conversion if not needed. in Chris function get_photometry so keep here.
            photo[filt] *= unyt_quantity.from_string("1 erg/s/Hz")
            photo[filt] = lnu_to_absolute_mag(photo[filt])
    return photo

def get_theta_SB(model="IllustrisTNG", device="cuda"):
    # theta is the number of simulation parameters so 28
    cam = camels(model=model, sim_set='SB28')
    theta = np.array([
        # commented out the others just to test first 
        cam.params['Omega0'].values,              # Omega0
        cam.params['sigma8'].values,              # sigma8
        cam.params['WindEnergyIn1e51erg'].values, # Wind Energy in 1e51 erg
        cam.params['RadioFeedbackFactor'].values,  # Radio Feedback Factor
        cam.params['VariableWindVelFactor'].values, # Variable Wind Velocity Factor
        cam.params['RadioFeedbackReiorientationFactor'].values, # Radio Feedback Reorientation Factor
        cam.params['OmegaBaryon'].values,         # Omega Baryon
        cam.params['HubbleParam'].values,         # Hubble Parameter
        cam.params['n_s'].values,                 # n_s
        cam.params['MaxSfrTimescale'].values,     # Max SFR Timescale
        cam.params['FactorForSofterEQS'].values,  # Factor for Softer EQS
        cam.params['IMFslope'].values,            # IMF slope
        cam.params['SNII_MinMass_Msun'].values,   # SNII Minimum Mass (Msun)
        cam.params['ThermalWindFraction'].values, # Thermal Wind Fraction
        cam.params['VariableWindSpecMomentum'].values, # Variable Wind Specific Momentum
        cam.params['WindFreeTravelDensFac'].values, # Wind Free Travel Density Factor
        cam.params['MinWindVel'].values,          # Minimum Wind Velocity
        cam.params['WindEnergyReductionFactor'].values, # Wind Energy Reduction Factor
        cam.params['WindEnergyReductionMetallicity'].values, # Wind Energy Reduction Metallicity
        cam.params['WindEnergyReductionExponent'].values, # Wind Energy Reduction Exponent
        cam.params['WindDumpFactor'].values,      # Wind Dump Factor
        cam.params['SeedBlackHoleMass'].values,   # Seed Black Hole Mass
        cam.params['BlackHoleAccretionFactor'].values, # Black Hole Accretion Factor
        cam.params['BlackHoleEddingtonFactor'].values, # Black Hole Eddington Factor
        cam.params['BlackHoleFeedbackFactor'].values, # Black Hole Feedback Factor
        cam.params['BlackHoleRadiativeEfficiency'].values, # Black Hole Radiative Efficiency
        cam.params['QuasarThreshold'].values,     # Quasar Threshold
        cam.params['QuasarThresholdPower'].values # Quasar Threshold Power
    ]).T
    
    return torch.tensor(theta, dtype=torch.float32, device=device)

def get_x_SB(
    spec_type="attenuated",
    snap="044",
    sps="BC03",
    luminosity_functions=True,
    colours=True,
    model="IllustrisTNG",
    photo_dir="/disk/xray15/aem2/data/28pams/IllustrisTNG/SB/photometry",
    n_bins_lf=13,
    n_bins_colour=13,
    uvlf_limits=(-24, -14),  # Default values from your config
    colour_limits=(-0.5, 3.5),  # Default values from your config
    filters=["GALEX FUV", "GALEX NUV"]
):
    """
    Get features (luminosity functions and/or colors) for SB28 simulations.
    
    Args:
        spec_type (str): Spectral type ("attenuated" by default)
        snap (str): Snapshot number
        sps (str): Stellar population synthesis model
        luminosity_functions (bool): Whether to compute luminosity functions
        colours (bool): Whether to compute color distributions
        model (str): Simulation model name
        photo_dir (str): Directory containing photometry data
        n_bins_lf (int): Number of bins for luminosity functions
        n_bins_colour (int): Number of bins for color distribution
        uvlf_limits (tuple): (low, high) magnitude limits for UVLF
        colour_limits (tuple): (low, high) limits for color distribution
        filters (list): List of filters to use
    
    Returns:
        list: Features for each simulation
    """
    if isinstance(snap, str):
        snap = [snap]

    x = [[] for _ in range(2048)]  # For SB28 simulations
    
    # Unpack limits
    uvlf_low, uvlf_high = uvlf_limits
    colour_low, colour_high = colour_limits

    for SB28_ in range(2048):
        try:
            for snp in snap:
                photo = get_photometry_SB(
                    sim_name=f"SB28_{SB28_}",
                    spec_type=spec_type,
                    snap=snp,
                    sps=sps,
                    model=model,
                    photo_dir=photo_dir,
                    filters=filters
                )

                if luminosity_functions:
                    # Create limits for each filter
                    filter_limits = [(filt, uvlf_low, uvlf_high) for filt in filters]
                    
                    for filt, lo_lim, hi_lim in filter_limits:
                        phi = get_luminosity_function(
                            photo, filt, lo_lim, hi_lim, n_bins=n_bins_lf
                        )[0]
                        x[SB28_].append(phi)

                if colours and len(filters) >= 2:
                    binLimsColour = np.linspace(colour_low, colour_high, n_bins_colour)
                    # Assuming we're always using the first two filters for color
                    color = photo[filters[0]] - photo[filters[1]]
                    color_dist = np.histogram(color, binLimsColour, density=True)[0]
                    x[SB28_].append(color_dist)

        except Exception as e:
            print(f"Error processing simulation {SB28_}: {e}")
            x[SB28_] = None

    # Remove any failed simulations
    x = [xi for xi in x if xi is not None]
    
    return x

# def get_x_SB( # get colours or LFs
#     # x.shape= (no. sims, no. bins*features) 

#         # 2 GALEX filters (FUV, NUV)
#         # Each filter gets n_bins_lf (12) bins
#         # 2 filters * 12 bins = 24 features
#         # 1 color (FUV-NUV)
#         # Gets n_bins_colour (9) bins
#         # 1 color * 9 bins = 9 features

#     # Total: 24 + 9 = 33 features per simulation with colours & UVLF but with just UVLF its 24
     
#     spec_type="attenuated",
#     snap="044",
#     sps="BC03",
#     luminosity_functions=True, # leave these as is, just overwrite in script if dont want both.
#     colours=True, 
#     model="IllustrisTNG",
#     photo_dir="/disk/xray15/aem2/data/28pams/IllustrisTNG/SB/photometry",
#     n_bins_lf=13, # 12 edges
#     n_bins_colour=13,
#     mag_limits=
#     uvlf_limits
# ):
#     if isinstance(snap, str):
#         snap = [snap]

#     x = [[] for _ in range(2048)]  # For SB28 simulations

#     for SB28_ in range(2048):
#         try:
#             for snp in snap:
#                 photo = get_photometry_SB(
#                     sim_name=f"SB28_{SB28_}",
#                     spec_type=spec_type,
#                     snap=snp,
#                     sps=sps,
#                     model=model,
#                     photo_dir=photo_dir,
#                 )

#                 if luminosity_functions:
#                     for filt, lo_lim, hi_lim in zip(
#                         ["GALEX FUV", "GALEX NUV"],# (-24, -16)
#                         [-24, -24], <  uvlf_limits  low,low # 24/25
#                         [-16, -16], <uvlf_limits , high  high# initial test. 
#                     ):
#                         phi = get_luminosity_function(
#                             photo, filt, lo_lim, hi_lim, n_bins=n_bins_lf
#                         )[0]
#                         x[SB28_].append(phi)

#                 if colours:
#                     binLimsColour = np.linspace(-0.5, 3.5, n_bins_colour)
#                     color = photo["GALEX FUV"] - photo["GALEX NUV"]
#                     color_dist = np.histogram(color, binLimsColour, density=True)[0]
#                     x[SB28_].append(color_dist)
#         except Exception as e:
#             print(f"Error processing simulation {SB28_}: {e}")
#             x[SB28_] = None

#     # Remove any failed simulations
#     x = [xi for xi in x if xi is not None]
    
#     return x


def get_theta_x_SB(
    spec_type="attenuated",
    snap="044",
    sps="BC03",
    model="IllustrisTNG",
    device="cuda",
    **kwargs,
):
    x = get_x_SB(spec_type=spec_type, snap=snap, sps=sps, model=model, **kwargs)
    theta = get_theta_SB(model=model, device=device)
    
    # Convert x list to proper array format
    x_array = np.array([np.hstack(_x) for _x in x])
    return theta, x_array


''' use:
if __name__ == "__main__":
    theta, x = get_theta_x_SB()
    print(theta.shape, x.shape)
'''
