"""
Create image example
====================

Example for generating a rest-frame physical scale image. This example will:
- Build a parametric galaxy (see make_sfzh and make_sed)
- Define its morphology
- Calculate rest-frame luminosities for the UVJ bands
- Make an image of the galaxy, including an RGB image.
"""


import matplotlib.pyplot as plt
from unyt import kpc, Myr

from synthesizer.filters import UVJ
from synthesizer.parametric.galaxy import Galaxy
from synthesizer.parametric import SFH, ZDist, Stars
from synthesizer.parametric.morphology import Sersic2D
from synthesizer.grid import Grid
from synthesizer.imaging.images import ParametricImage


if __name__ == "__main__":
    # Define the morphology using a simple effective radius and slope
    morph = Sersic2D(r_eff=1 * kpc, sersic_index=1.0, ellipticity=0.5)

    # Get the location of this script, __file__ is the absolute path of this
    # script, however we just want to directory
    # script_path = os.path.abspath(os.path.dirname(__file__))

    # Define the grid
    grid_name = "test_grid"
    grid_dir = "../../tests/test_grid/"
    grid = Grid(grid_name, grid_dir=grid_dir)

    # Define the SFZDist
    Z_p = {"metallicity": 0.01}
    metal_dist = ZDist.DeltaConstant(**Z_p)
    sfh_p = {"duration": 100 * Myr}
    sfh = SFH.Constant(**sfh_p)  # constant star formation
    sfzh = Stars(
        grid.log10age,
        grid.metallicity,
        sf_hist=sfh,
        metal_dist=metal_dist,
        initial_mass=10**9,
        morphology=morph,
    )

    # Initialise a parametric Galaxy
    galaxy = Galaxy(sfzh)

    # Generate stellar spectra
    galaxy.stars.get_spectra_incident(grid)

    # Get a UVJ filter set
    filters = UVJ()

    # Define geometry of the images
    resolution = 0.05 * kpc  # resolution in kpc
    npix = 50
    fov = resolution.value * npix * kpc

    # Generate images using the low level image methods
    img = ParametricImage(
        morphology=morph,
        resolution=resolution,
        filters=filters,
        sed=galaxy.stars.spectra["incident"],
        fov=fov,
    )

    # Get the photometric images
    img.get_imgs()

    # Make and plot an rgb image
    img.make_rgb_image(rgb_filters={"R": "J", "G": "V", "B": "U"})
    fig, ax, _ = img.plot_rgb_image()

    plt.show()

    # We can also do the same with a helper function on the galaxy object
    img = galaxy.make_images(
        resolution=resolution,
        filters=filters,
        stellar_spectra_type="incident",
        fov=fov,
    )

    # and... print an ASCII representation
    img.print_ascii(filter_code="U")
