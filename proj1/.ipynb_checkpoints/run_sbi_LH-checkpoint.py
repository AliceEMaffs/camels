import sys

sys.path.insert(0, "..")

import torch

torch.set_default_dtype(torch.float32)

import numpy as np
import ili
from ili.dataloaders import NumpyLoader
from ili.inference import InferenceRunner
from ili.validation.metrics import PosteriorCoverage
from sklearn.preprocessing import Normalizer
import joblib

from priors import initialise_priors
from setup_params import get_theta_x
from camels import camels


# IllustrisTNG_all_BC03_attenuated_12_12_086

model = "IllustrisTNG"  # "Swift-EAGLE" # "Astrid" # "IllustrisTNG" # "Simba"
spec_type = "intrinsic"
sps = "BC03"
snap = ["086"]  # , "060", "044"] #  "060", "044"]  # , "086", "060", "044"]
n_bins_lf = 12 
n_bins_colour = 12
cam = camels(model)

bands = "all"
colours = True
luminosity_functions = True

name = f"{model}_{bands}_{sps}_{spec_type}_{n_bins_lf}_{n_bins_colour}"

if isinstance(snap, list):
    for snp in snap:
        name += f"_{snp}"
else:
    name += f"_{snap}"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

prior = initialise_priors(device=device, astro=True, dust=False)
theta, x = get_theta_x(
    # photo_dir=f"/mnt/ceph/users/clovell/CAMELS_photometry/{model}/",
    photo_dir=f"/mnt/ceph/users/camels/PUBLIC_RELEASE/Photometry/{model}/L25n256/LH/",
    spec_type=spec_type,
    model=model,
    snap=snap,
    sps=sps,
    n_bins_lf=n_bins_lf,
    n_bins_colour=n_bins_colour,
    colours=colours,
    luminosity_functions=luminosity_functions,
    device=device,
)

x_all = np.array([np.hstack(_x) for _x in x])

# # Make sure no constant variables, to avoid nan loss with lampe NDE models
# x_all[x_all == 0.0] = np.array(
#     np.random.rand(np.sum((x_all == 0.0))) * 1e-10
# )

norm = Normalizer()
x_all = torch.tensor(
    norm.fit_transform(X=x_all),
    # x_all,
    dtype=torch.float32,
    device=device, 
)

joblib.dump(norm, f'models/{name}_scaler.save')


# test_mask = np.random.rand(1000) > 0.9
# np.savetxt('../data/test_mask.txt', test_mask, fmt='%i')
test_mask = np.loadtxt("../data/test_mask.txt", dtype=bool)

hidden_features = 30
num_transforms = 4
nets = [
    # ili.utils.load_nde_sbi(
    #     engine="NLE", model="maf", hidden_features=50, num_transforms=5
    # ),
    ili.utils.load_nde_sbi(
        engine="NPE",
        model="nsf", hidden_features=hidden_features, num_transforms=num_transforms
    ),
    ili.utils.load_nde_sbi(
        engine="NPE",
        model="nsf", hidden_features=hidden_features, num_transforms=num_transforms
    ),
    ili.utils.load_nde_sbi(
        engine="NPE",
        model="nsf", hidden_features=hidden_features, num_transforms=num_transforms
    ),
    # ili.utils.load_nde_sbi(
    #     engine="NPE",
    #     model="nsf", hidden_features=hidden_features, num_transforms=num_transforms
    # ),
    # ili.utils.load_nde_lampe(model="nsf", device=device, hidden_features=20, num_transforms=2), 
    # ili.utils.load_nde_lampe(model="nsf", device=device, hidden_features=20, num_transforms=2), 
]

train_args = {"training_batch_size": 4, "learning_rate": 5e-4, 'stop_after_epochs': 20}

loader = NumpyLoader(
    x=x_all[~test_mask],
    # theta=torch.tensor(theta[~test_mask], device=device)
    theta=torch.tensor(theta[~test_mask, :], device=device)
)

runner = InferenceRunner.load(
    backend="sbi",  #'sbi', # 'lampe',
    engine="NPE",
    prior=prior,
    nets=nets,
    device=device,
    train_args=train_args,
    proposal=None,
    # embedding_net=None,
    out_dir="models/",
    name=name,
)

posterior_ensemble, summaries = runner(loader=loader)


"""
Coverage plots for each model
"""
metric = PosteriorCoverage(
    num_samples=int(4e3),
    sample_method='direct',
    # sample_method="slice_np_vectorized",
    # sample_params={'num_chains': 1},
    # sample_method="vi",
    # sample_params={"dist": "maf", "n_particles": 32, "learning_rate": 1e-2},
    labels=cam.labels,
    plot_list=["coverage", "histogram", "predictions", "tarp"],
    # out_dir="../plots/",
)

fig = metric(
    posterior=posterior_ensemble,
    x=x_all[test_mask].cpu(),
    theta=theta[test_mask, :].cpu(),
    signature=f"coverage_{name}_",
)

fig[3].axes[0].set_xlim(0,1)
fig[3].axes[0].set_ylim(0,1)
fig[3].savefig(f'../plots/coverage_{name}_plot_TARP.png', bbox_inches='tight', dpi=200)

