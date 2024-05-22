"""
Dust curves example
===================

Plot dust curves
"""

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from synthesizer.dust import attenuation
from unyt import Angstrom

models = [
    "PowerLaw",
    "Calzetti2000",
    "MWN18",
    "GrainsWD01",
    "GrainsWD01",
    "GrainsWD01",
    "ParametricLI08",
    "ParametricLI08",
    "ParametricLI08",
    "ParametricLI08",
]

params = [
    {"slope": -1.0},
    {"slope": 0.0, "cent_lam": 0.2175, "ampl": 1.26, "gamma": 0.0356},
    {},
    {"model": "MW"},
    {"model": "SMC"},
    {"model": "LMC"},
    {"model": "MW"},
    {"model": "LMC"},
    {"model": "SMC"},
    {"model": "Calzetti"},
]

colors = cmr.take_cmap_colors("cmr.guppy", len(models))

lam = np.arange(1000, 10000, 10) * Angstrom

for ii, (model, param) in enumerate(zip(models, params)):
    emodel = getattr(attenuation, model)(**param)

    plt.plot(
        lam, emodel.get_tau(lam), color=colors[ii], label=f"{model}, {param}"
    )

plt.xlabel(r"$\lambda/(\AA)$", fontsize=12)
plt.ylabel(r"A$_{\lambda}/$A$_{V}$", fontsize=12)
plt.yticks(np.arange(0, 10))
plt.xlim(np.min(lam), np.max(lam))

plt.legend(frameon=False, fontsize=10)
plt.grid()

plt.show()
