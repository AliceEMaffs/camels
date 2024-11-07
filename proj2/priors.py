import torch
from torch.distributions import Uniform, TransformedDistribution, ExpTransform
from sbi.utils.user_input_checks import process_prior


def initialise_priors(device="cuda", astro=True, dust=True):

    combined_priors = []

    if astro:
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
