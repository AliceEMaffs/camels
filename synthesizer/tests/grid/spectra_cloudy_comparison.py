import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from synthesizer.grid import Grid


sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)


def plot_spectra(grid, log10Z=-2.0, log10age=6.0, spec_names=None):
    return fig, ax


if __name__ == "__main__":
    """
    Define choise of SPS model and initial mass function (IMF)
    """
    sps_names = [
        "bpass-v2.2.1-bin_chab-100_cloudy-v17.03_log10Uref-2",
        "bpass-v2.2.1-bin_chab-100_cloudy-v17.00_log10Uref-2",
    ]

    log10Z = -2.0  # log10(metallicity)
    log10age = 6.0  # log10(age/yr)

    spec_names = ["total"]

    grid1 = Grid(sps_names[0])
    grid2 = Grid(sps_names[1])

    iZ, log10Z = grid1.get_nearest_log10Z(log10Z)
    print(f"closest metallicity: {log10Z:.2f}")
    ia, log10age = grid1.get_nearest_log10age(log10age)
    print(f"closest age: {log10age:.2f}")

    fig = plt.figure(figsize=(3.5, 5.0))

    left = 0.2
    height = 0.8
    bottom = 0.1
    width = 0.75

    ax = fig.add_axes((left, bottom, width, height))

    ax.axhline(c="k", lw=3, alpha=0.05)

    for spec_name in spec_names:
        Lnu1 = grid1.spectra[spec_name][ia, iZ, :]
        Lnu2 = grid2.spectra[spec_name][ia, iZ, :]

        ax.plot(
            np.log10(grid1.lam),
            np.log10(Lnu2 / Lnu1),
            lw=1,
            alpha=0.8,
            label=spec_name,
        )

    ax.set_xlim([3.0, 4.0])
    ax.set_ylim([-0.75, 0.75])
    ax.legend(fontsize=8, labelspacing=0.0)
    ax.set_xlabel(r"$\rm log_{10}(\lambda/\AA)$")
    ax.set_ylabel(r"$\rm log_{10}(L_{\nu}^2/L_{\nu}^1)$")

    fig.savefig("figs/comparison.pdf")
