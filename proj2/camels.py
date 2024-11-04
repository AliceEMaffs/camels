import time
import numpy as np
import pandas as pd
from astropy.cosmology import Planck15, Planck18

from unyt import Myr
from synthesizer.emission_models import EmissionModel, IncidentEmission
from synthesizer.filters import Filter, FilterCollection
from synthesizer.emission_models.attenuation import ParametricLi08
from synthesizer.filters import Filter, FilterCollection
from synthesizer.load_data.load_camels import (
    load_CAMELS_Simba,
    load_CAMELS_IllustrisTNG,
    load_CAMELS_Astrid,
    load_CAMELS_SwiftEAGLE_subfind,
)


class camels:
    def __init__(self, model="IllustrisTNG", sim_set='LH', extended=False):
        self.model = model
        self.h = 0.6711
        self.sim_set = sim_set
        self.extended = extended

        if self.sim_set == 'SB28':
            if self.model not in ['IllustrisTNG']:
                raise ValueError(f'SB28 set not available for {self.model} model')

        if ((self.sim_set in ['1P']) & self.extended):
            if self.model not in ['IllustrisTNG', 'Simba']:
                raise ValueError(f'Extended 1P parameter set not available for {self.model} model')

        if self.model == "Simba":
            self.fname = (
                "/mnt/ceph/users/camels/PUBLIC_RELEASE/Sims/SIMBA/L25n256/"
                f"{self.sim_set}/CosmoAstroSeed_SIMBA_L25n256_{self.sim_set}.txt"
            )
            self.sim_method = load_CAMELS_Simba
            self.cosmology = Planck15
        elif self.model == "Astrid":
            self.fname = (
                "/mnt/ceph/users/camels/PUBLIC_RELEASE/Sims/Astrid/L25n256/"
                f"{self.sim_set}/CosmoAstroSeed_Astrid_L25n256_{self.sim_set}.txt"
            )
            self.sim_method = load_CAMELS_Astrid
            self.cosmology = Planck18
        elif self.model == "IllustrisTNG":
            self.fname = (
                "/mnt/ceph/users/camels/PUBLIC_RELEASE/Sims/IllustrisTNG/L25n256/"
                f"{self.sim_set}/CosmoAstroSeed_IllustrisTNG_L25n256_{self.sim_set}.txt"
            )
            self.sim_method = load_CAMELS_IllustrisTNG
            self.cosmology = Planck18
        elif self.model == "Swift-EAGLE":
            self.fname = (
                "/mnt/home/clovell/code/camels_observational_catalogues/"
                "CosmoAstroSeed_SwiftEAGLE_L25n256_LH.txt"
            )
            self.sim_method = load_CAMELS_SwiftEAGLE_subfind
            self.cosmology = Planck18
        elif self.model == "Arbitrary":
            self.fname = None
            self.sim_method = None
            self.cosmology = None
        else:
            raise ValueError("`sim` not recognised")
        
        self.LH = np.arange(1000)

        if self.fname is not None:
            self.params = pd.read_table(self.fname, delim_whitespace=True)

        self.define_sims()
        self.define_param_labels()

        self.snaps = np.array(
            [
                "014",
                "018",
                "024",
                "028",
                "032",
                "034",
                "036",
                "038",
                "040",
                "042",
                "044",
                "046",
                "048",
                "050",
                "052",
                "054",
                "056",
                "058",
                "060",
                "062",
                "064",
                "066",
                "068",
                "070",
                "072",
                "074",
                "076",
                "078",
                "080",
                "082",
                "084",
                "086",
                "088",
                "090",
            ]
        )

        self.redshifts = np.array(
            [
                6.01,
                5.00,
                4.01,
                3.49,
                3.01,
                2.80,
                2.63,
                2.46,
                2.30,
                2.15,
                2.00,
                1.86,
                1.73,
                1.60,
                1.48,
                1.37,
                1.26,
                1.14,
                1.05,
                0.95,
                0.86,
                0.77,
                0.69,
                0.61,
                0.54,
                0.47,
                0.40,
                0.34,
                0.27,
                0.21,
                0.15,
                0.10,
                0.05,
                0.00,
            ]
        )

    def define_sims(self):
        sim_names = []
        
        if self.sim_set == 'LH':         
            sim_names += [f'LH_{i}' for i in range(1000)]

        if self.sim_set == 'CV':
            sim_names += [f'CV_{i}' for i in range(27)]

        if self.sim_set == 'SB28':
            sim_names += [f'SB28_{i}' for i in range(2048)]

        if self.sim_set == '1P':
            _names = [[f'1P_p{k}_{j}' for j in ['2', '1', 'n1', 'n2']] for k in range(1, 7)]
            sim_names += [x for xs in _names for x in xs]  # flatten list of lists
            sim_names += ['1P_p1_0']
            
            # Add more 1P sims from 28 parameter set
            if self.extended:
                extra_1p_sims = [
                    [
                        f'1P_p{k}_{j}' for j in ['2', '1', 'n1', 'n2']
                    ] for k in range(7, 29) if k not in [15]
                ]

                sim_names += [x for xs in extra_1p_sims for x in xs]
    
        self.sim_names = sim_names
        

    def define_param_labels(self):
        if self.sim_set in ['CV', '1P', 'LH']:
                self.labels = [
                    r"$\Omega_m$",
                    r"$\sigma_8$",
                    r"$A_{\mathrm{SN1}}$",
                    r"$A_{\mathrm{AGN1}}$",
                    r"$A_{\mathrm{SN2}}$",
                    r"$A_{\mathrm{AGN2}}$",
                ]

        if (self.sim_set == 'SB28') | ((self.sim_set == '1P') & self.extended):
            if self.model == 'IllustrisTNG':
                self.labels = [
                    'Omega0', 'sigma8', 'WindEnergyIn1e51erg',
                    'RadioFeedbackFactor', 'VariableWindVelFactor',
                    'RadioFeedbackReiorientationFactor', 'OmegaBaryon', 'HubbleParam',
                    'n_s', 'MaxSfrTimescale', 'FactorForSofterEQS', 'IMFslope',
                    'SNII_MinMass_Msun', 'ThermalWindFraction', 'VariableWindSpecMomentum',
                    'WindFreeTravelDensFac', 'MinWindVel', 'WindEnergyReductionFactor',
                    'WindEnergyReductionMetallicity', 'WindEnergyReductionExponent',
                    'WindDumpFactor', 'SeedBlackHoleMass', 'BlackHoleAccretionFactor',
                    'BlackHoleEddingtonFactor', 'BlackHoleFeedbackFactor',
                    'BlackHoleRadiativeEfficiency', 'QuasarThreshold',
                    'QuasarThresholdPower',
                ]


        if ((self.sim_set == '1P') & self.extended):
            if self.model == 'Simba':
                self.labels = [
                    'Omega0', 'sigma8', 'GALSF_SUBGRID_DAA_LOADFACTOR', 
                    'BH_BAL_KICK_MOMENTUM_FLUX', 'GALSF_SUBGRID_FIREVEL', 
                    'BH_QUENCH_JET', 'OmegaBaryon', 'HubbleParam', 'ns', 
                    'CritPhysDensity', 'SfEffPerFreeFall', 'TempClouds', 
                    'WindFreeTravelMaxTime', 'WindFreeTravelDensFac', 
                    'SeedBlackHoleMass', 'BlackHoleAccretionFactor', 
                    'BlackHoleRadiativeEfficiency', 'BlackHoleEddingtonFactor', 
                    'BlackHoleNgbFactor', 'BlackHoleMaxAccretionRadius', 'GALSF_JEANS_MIN_T', 
                    'GALSF_SUBGRID_FIREVEL_SLOPE', 'GALSF_SUBGRID_HOTWIND',
                    'GALSF_AGBWINDHEATING', 'BH_HOST_TO_SEED_RATIO', 'BH_QUENCH_JET_HOTWIND', 
                    'BH_BONDI_HOT', 'MBH_JET_MIN'
                ]


    def define_filter_collection(self, lam=None, filter_directory="data/"):
        fs = [f"SLOAN/SDSS.{f}" for f in ["u", "g", "r", "i", "z"]]
        fs += [f"Generic/Johnson.{f}" for f in ["U", "B", "V", "J"]]
        fs += [f"UKIRT/UKIDSS.{f}" for f in ["Y", "J", "H", "K"]]
        fs += [
            f"HST/ACS_HRC.{f}" for f in ["F435W", "F606W", "F775W", "F814W", "F850LP"]
        ]
        fs += [
            f"HST/WFC3_IR.{f}"
            for f in ["F098M", "F105W", "F110W", "F125W", "F140W", "F160W"]
        ]
        fs += [
            f"JWST/NIRCam.{f}"
            for f in [
                "F070W",
                "F090W",
                "F115W",
                "F150W",
                "F200W",
                "F277W",
                "F356W",
                "F444W",
            ]
        ]

        tophats = {
            "UV1500": {"lam_eff": 1500, "lam_fwhm": 300},
            "UV2800": {"lam_eff": 2800, "lam_fwhm": 300},
        }

        flam, trans = np.loadtxt(f"{filter_directory}/galex1500.res").T
        galex1500 = Filter("GALEX FUV", transmission=trans, new_lam=flam)

        flam, trans = np.loadtxt(f"{filter_directory}/galex2500.res").T
        galex2500 = Filter("GALEX NUV", transmission=trans, new_lam=flam)

        self.fc = FilterCollection(
            filter_codes=fs,
            tophat_dict=tophats,
            filters=[galex1500, galex2500],
            new_lam=lam,
            verbose=False,
        )

        return self.fc

    def get_galaxies(self, sim, snap, mstar_lim=8, nthreads=1):
        if self.model == "Simba":
            _model = "SIMBA"
        else:
            _model = self.model

        gals = self.sim_method(
            (
                f"/mnt/ceph/users/camels/PUBLIC_RELEASE/Sims/{_model}/"
                f"L25n256/{self.sim_set}/{sim}"
            ),
            snap_name=f"snapshot_{snap}.hdf5",
            group_name=f"groups_{snap}.hdf5",
            group_dir=(
                f"/mnt/ceph/users/camels/PUBLIC_RELEASE/FOF_Subfind/{_model}/"
                f"L25n256/{self.sim_set}/{sim}"
            ),
            num_threads=nthreads,
        )

        # Filter by stellar mass
        mstar = np.array([np.sum(_g.stars.current_masses) for _g in gals])
        mstar[mstar <= 0.0] = 1
        mstar = np.log10(mstar)
        mask = np.where(mstar > mstar_lim)[0]  # < 10 star particles...
        gals = [g for i, g in enumerate(gals) if i in mask]

        return gals, mask

    def define_emission_model(
        self,
        grid,
        params=None,
        dust_model=None,
        save_extra=False,
    ):
        if params is None:
            params = {
                "UV_slope": None,
                "OPT_NIR_slope": None,
                "FUV_slope": None,
                "bump": None,
            }

        dust_curve = ParametricLi08(
            UV_slope=params["UV_slope"],
            OPT_NIR_slope=params["OPT_NIR_slope"],
            FUV_slope=params["FUV_slope"],
            bump=params["bump"],
            model=dust_model,
        )

        young_incident = IncidentEmission(
            grid=grid,
            mask_attr="ages",
            mask_op="<",
            mask_thresh=10 * Myr,
            save=save_extra,
            emitter="stellar",
        )

        young_emergent = EmissionModel(
            "attenuated_young",
            dust_curve=dust_curve,
            apply_dust_to=young_incident,
            tau_v=0.67,
            save=save_extra,
            emitter="stellar",
        )

        old_incident = EmissionModel(
            "old_incident",
            grid=grid,
            extract="incident",
            mask_attr="ages",
            mask_op=">",
            mask_thresh=10 * Myr,
            save=save_extra,
            emitter="stellar",
        )

        total_intrinsic = EmissionModel(
            "intrinsic",
            combine=(old_incident, young_incident),
            emitter="stellar",
        )

        old_and_young = EmissionModel(
            "old_intrinsic_young_attenuated",
            combine=(old_incident, young_emergent),
            emitter="stellar",
            save=save_extra,
        )

        total = EmissionModel(
            "attenuated",
            dust_curve=dust_curve,
            apply_dust_to=old_and_young,
            tau_v=0.33,
            emitter="stellar",
            related_models=total_intrinsic,
        )

        return total

    def get_photometry(
        self,
        gals,
        grid,
        fc,
        # filter_directory="data/",
        params=None,
        dust_model=None,
        save_extra=False,
    ):
        emodel = self.define_emission_model(
            grid=grid,
            params=params,
            dust_model=dust_model,
            save_extra=save_extra,
        )

        # Use parametric young stars
        # [
        #     g.stars.parametric_young_stars(
        #         age=100 * Myr,
        #         parametric_sfh='constant',
        #         grid=grid,
        #     ) for i, g in enumerate(gals)
        # ]

        start = time.time()
        for _g in gals:
            # Get rest-frame spectra
            _g.stars.get_spectra(
                emodel,
                # aperture=30 * kpc,
            )

            # Get observer-frame spectra
            _g.get_observed_spectra(cosmo=self.cosmology)

        print(f"Spectra generation took: {time.time() - start:.2f}")

        start = time.time()
        for _g in gals:
            _g.get_photo_lnu(fc)
            _g.get_photo_fnu(fc)

        print(f"Photometry generation took: {time.time() - start:.2f}")

        return emodel

    def calc_df(self, _x, volume, massBinLimits):
        hist, dummy = np.histogram(_x, bins=massBinLimits)
        hist = np.float64(hist)
        phi = (hist / volume) / (massBinLimits[1] - massBinLimits[0])

        phi_sigma = (np.sqrt(hist) / volume) / (
            massBinLimits[1] - massBinLimits[0]
        )  # Poisson errors

        return phi, phi_sigma, hist