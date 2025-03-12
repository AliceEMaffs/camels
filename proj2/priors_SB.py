import torch
import pandas as pd
import numpy as np
from torch.distributions import Uniform, TransformedDistribution, ExpTransform
from sbi.utils.user_input_checks import process_prior

# parameters defined here: /disk/xray15/aem2/data/28pams/CosmoAstroSeed_IllustrisTNG_L25n256_SB28.txt

''' priors from looking at their shapes:

uniform/linear
omega0 = [0.1,0.5], fid = 0.3
sigma8 = [0.6, 1], fid = 0.8
OmegaBaryon = [0.029,0.069], fid = 0.049
HubbleParam
n_s
IMFslope
SNII_MinMass_Msun
VariableWindSpecMomentum
MinWindVel
WindEnergyReductionExponent
WindDumpFactor
QuasarThresholdPower

exponential/logarithmic
WindEnergyIn1e51erg = [0.9,14.4], fid =3.6
RadioFeedbackFactor = [0.25,4], fid = 1.0
VariableWindVelFactor = [3.7,14.8], fid=7.4
RadioFeedbackReiorientationFactor = [10,40], fid = 20
MaxSfrTimescale
FactorForSofterEQS
ThermalWindFraction
WindFreeTravelDensFac
WindEnergyReductionFactor
WindEnergyReductionMetallicity
SeedBlackHoleMass
BlackHoleAccretionFactor
BlackHoleEddingtonFactor
BlackHoleFeedbackFactor
BlackHoleRadiativeEfficiency
QuasarThreshold


'''
def initialise_priors_LH(device="cuda", astro=True, dust=True):

    combined_priors = []

    if astro:
        # need to define the ranges of each of the priors.
        base_dist1 = Uniform(
            torch.log(torch.tensor([0.25], device=device)),
            torch.log(torch.tensor([4], device=device)),
        )
        base_dist2 = Uniform(
            torch.log(torch.tensor([0.5], device=device)),
            torch.log(torch.tensor([2], device=device)),
        )
        astro_prior1 = TransformedDistribution(base_dist1, ExpTransform())
        astro_prior2 = TransformedDistribution(base_dist2, ExpTransform())
        omega_prior = Uniform(
            torch.tensor([0.1], device=device),
            torch.tensor([0.5], device=device),
        )
        sigma8_prior = Uniform(
            torch.tensor([0.6], device=device),
            torch.tensor([1.0], device=device),
        )
        combined_priors += [
            omega_prior,
            sigma8_prior,
            astro_prior1,
            astro_prior2,
            astro_prior1,
            astro_prior2,
        ]

    if dust:
        tau_ism_prior = Uniform(
            torch.tensor([0.1], device=device),
            torch.tensor([0.5], device=device),
        )
        tau_bc_prior = Uniform(
            torch.tensor([0.5], device=device),
            torch.tensor([1.0], device=device),
        )
        uv_slope_prior = Uniform(
            torch.tensor([2.0], device=device),
            torch.tensor([22.0], device=device),
        )
        opt_nir_slope_prior = Uniform(
            torch.tensor([1.0], device=device),
            torch.tensor([6.0], device=device),
        )
        fuv_slope_prior = Uniform(
            torch.tensor([-1.0], device=device),
            torch.tensor([2.0], device=device),
        )
        bump_prior = Uniform(
            torch.tensor([0.0], device=device),
            torch.tensor([0.2], device=device),
        )

        combined_priors += [
            # tau_ism_prior,
            # tau_bc_prior,
            uv_slope_prior,
            opt_nir_slope_prior,
            fuv_slope_prior,
            bump_prior,
        ]

    prior = process_prior(combined_priors)

    return prior[0]

import torch
from torch.distributions import Uniform, TransformedDistribution
from torch.distributions.transforms import ExpTransform

def initialise_priors_SB28(df, astro=True, dust=True, device="cuda"): 
    
    # relies on reading in the priors from /disk/xray15/aem2/data/28pams/Info_IllustrisTNG_L25n256_28params.txt into df
    # this contains info on log, min, max which is needed to define prior.
    combined_priors = []

    if astro:
        for index, row in df.iterrows(): # for all 28 parameters in the dataframe
            param_name = row['ParamName']
            log_flag = int(row['LogFlag'])
            min_val = row['MinVal']
            max_val = row['MaxVal']
            #fiducial_val = row['FiducialVal']
            print(f"processing {param_name}:")
            # Define the prior based on LogFlag
            if log_flag == 1:  # Logarithmic prior
                print(f"processing {param_name} is logarithmic with min {min_val} and max {max_val}:")
                base_dist = Uniform(
                    torch.log(torch.tensor([min_val], device=device)),
                    torch.log(torch.tensor([max_val], device=device)),
                )
                prior = TransformedDistribution(base_dist, ExpTransform())
            else:  # Linear prior
                print(f"processing {param_name} is linear with min {min_val} and max {max_val}:")
                prior = Uniform(
                    torch.tensor([min_val], device=device),
                    torch.tensor([max_val], device=device),
                )
            
            combined_priors.append(prior)
        
    if dust:
        tau_ism_prior = Uniform(
            torch.tensor([0.1], device=device),
            torch.tensor([0.5], device=device),
        )
        tau_bc_prior = Uniform(
            torch.tensor([0.5], device=device),
            torch.tensor([1.0], device=device),
        )
        uv_slope_prior = Uniform(
            torch.tensor([2.0], device=device),
            torch.tensor([22.0], device=device),
        )
        opt_nir_slope_prior = Uniform(
            torch.tensor([1.0], device=device),
            torch.tensor([6.0], device=device),
        )
        fuv_slope_prior = Uniform(
            torch.tensor([-1.0], device=device),
            torch.tensor([2.0], device=device),
        )
        bump_prior = Uniform(
            torch.tensor([0.0], device=device),
            torch.tensor([0.2], device=device),
        )

        combined_priors += [
            # tau_ism_prior,
            # tau_bc_prior,
            uv_slope_prior,
            opt_nir_slope_prior,
            fuv_slope_prior,
            bump_prior,
        ]    

    prior = process_prior(combined_priors)

    return prior[0]


def initialise_priors_SB28_splitGPU(df, astro=True, dust=True, device="cuda"): 
    """
    Initialize priors following SBI's requirements for distribution handling.
    The key is to create all distributions properly before combining them.
    """
    print(f"Initializing priors for eventual use on {device}")
    
    def create_single_prior(min_val, max_val, log=False):
        """Helper to create a single prior distribution."""
        if log:
            # For log-distributed parameters
            base_dist = Uniform(
                torch.log(torch.tensor([min_val], device=device)),
                torch.log(torch.tensor([max_val], device=device))
            )
            return TransformedDistribution(base_dist, ExpTransform())
        else:
            # For linearly-distributed parameters
            return Uniform(
                torch.tensor([min_val], device=device),
                torch.tensor([max_val], device=device)
            )

    # List to hold all individual priors
    individual_priors = []

    if astro:
        for index, row in df.iterrows():
            param_name = row['ParamName']
            log_flag = int(row['LogFlag'])
            min_val = float(row['MinVal'])
            max_val = float(row['MaxVal'])
            
            print(f"Creating prior for {param_name}")
            prior = create_single_prior(min_val, max_val, log=log_flag == 1)
            individual_priors.append(prior)
    
    # Let SBI handle the combination of priors
    print("Processing priors through SBI...")
    processed_prior = process_prior(individual_priors)[0]
    
    return processed_prior