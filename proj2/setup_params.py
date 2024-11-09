import sys

sys.path.insert(0, "../")

import torch

torch.set_default_dtype(torch.float32)

import numpy as np
import h5py
from unyt import unyt_quantity
from synthesizer.conversions import lnu_to_absolute_mag
from camels import camels


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
            cam.omegam,
            cam.sigma8,
            cam.A_SN1,
            cam.A_AGN1,
            cam.A_SN2,
            cam.A_AGN2,
            # dat.tau_ism,
            # dat.tau_bc,
            # dat.UV_slope,
            # dat.OPT_NIR_slope,
            # dat.FUV_slope,
            # dat.bump,
        ]
    ).T

    return torch.tensor(theta, dtype=torch.float32, device=device)


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


def get_x(
    spec_type="attenuated",
    snap="090",
    sps="BC03",
    luminosity_functions=True,
    colours=True,
    model="IllustrisTNG",
    photo_dir=("/disk/xray15/aem2/data/28pams/IllustrisTNG/photometry"),
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
                    color_dist = np.histogram(color, binLimsColour, density=True)[0]

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


if __name__ == "__main__":
    theta, x = get_theta_x()
    print(theta.shape, x.shape)
