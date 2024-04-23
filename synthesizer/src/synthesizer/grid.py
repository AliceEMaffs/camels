import os
import numpy as np
import h5py

import cmasher as cmr
import matplotlib as mpl
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from spectres import spectres
from unyt import unyt_array, unyt_quantity

from synthesizer import exceptions
from synthesizer.sed import Sed
from synthesizer.line import Line, LineCollection
from synthesizer.units import Quantity

from . import __file__ as filepath


def check_lines_available(grid_name, grid_dir):
    """
    Check that lines are available on this grid

    Args:
        grid_name (str):
            The name of the grid file.
        grid_dir (str):
            The directory to the grid file.
    """

    grid_filename = f"{grid_dir}/{grid_name}.hdf5"
    with h5py.File(grid_filename, "r") as hf:
        if "lines" in hf.keys():
            return True
        else:
            return False


def get_available_lines(grid_name, grid_dir, include_wavelengths=False):
    """
    Get a list of the lines available to a grid

    Args:
        grid_name (str):
            The name of the grid file.
        grid_dir (str):
            The directory to the grid file.

    Returns:
        lines (list):
            List of available lines
        wavelengths (list)
            List of associated wavelengths (if `include_wavelengths` is True)
    """

    grid_filename = f"{grid_dir}/{grid_name}.hdf5"
    with h5py.File(grid_filename, "r") as hf:
        lines = list(hf["lines"].keys())

        if include_wavelengths:
            wavelengths = np.array(
                [hf["lines"][line].attrs["wavelength"] for line in lines]
            )
            return lines, wavelengths
        else:
            return lines


def flatten_linelist(list_to_flatten):
    """
    Flatten a mixed list of lists and strings and remove duplicates. Used when
    converting a desired line list which may contain single lines and doublets.

    Args:
        list_to_flatten (list)
            list containing lists and/or strings and integers

    Returns:
        (list)
            flattened list
    """

    flattened_list = []
    for lst in list_to_flatten:
        if isinstance(lst, list) or isinstance(lst, tuple):
            for ll in lst:
                flattened_list.append(ll)

        elif isinstance(lst, str):
            # If the line is a doublet, resolve it and add each line
            # individually
            if len(lst.split(",")) > 1:
                flattened_list += lst.split(",")
            else:
                flattened_list.append(lst)

        else:
            raise Exception(
                (
                    "Unrecognised type provided. Please provide"
                    "a list of lists and strings"
                )
            )

    return list(set(flattened_list))


def parse_grid_id(grid_id):
    """
    This is used for parsing a grid ID to return the SPS model,
    version, and IMF

    Args:
        grid_id (str)
            string grid identifier
    """

    if len(grid_id.split("_")) == 2:
        sps_model_, imf_ = grid_id.split("_")
        cloudy = cloudy_model = ""

    if len(grid_id.split("_")) == 4:
        sps_model_, imf_, cloudy, cloudy_model = grid_id.split("_")

    if len(sps_model_.split("-")) == 1:
        sps_model = sps_model_.split("-")[0]
        sps_model_version = ""

    if len(sps_model_.split("-")) == 2:
        sps_model = sps_model_.split("-")[0]
        sps_model_version = sps_model_.split("-")[1]

    if len(sps_model_.split("-")) > 2:
        sps_model = sps_model_.split("-")[0]
        sps_model_version = "-".join(sps_model_.split("-")[1:])

    if len(imf_.split("-")) == 1:
        imf = imf_.split("-")[0]
        imf_hmc = ""

    if len(imf_.split("-")) == 2:
        imf = imf_.split("-")[0]
        imf_hmc = imf_.split("-")[1]

    if imf in ["chab", "chabrier03", "Chabrier03"]:
        imf = "Chabrier (2003)"
    if imf in ["kroupa"]:
        imf = "Kroupa (2003)"
    if imf in ["salpeter", "135all"]:
        imf = "Salpeter (1955)"
    if imf.isnumeric():
        imf = rf"$\alpha={float(imf)/100}$"

    return {
        "sps_model": sps_model,
        "sps_model_version": sps_model_version,
        "imf": imf,
        "imf_hmc": imf_hmc,
    }


class Grid:
    """
    The Grid class, containing attributes and methods for reading and
    manipulating spectral grids.

    Attributes:
        grid_dir (str)
            The directory containing the grid HDF5 file.
        grid_name (str)
            The name of the grid (as defined by the file name)
            with no extension.
        grid_ext (str)
            The grid extension. Either ".hdf5" or ".h5". If the passed
            grid_name has no extension then ".hdf5" is assumed.
        read_lines (bool/list)
            Flag for whether to read lines. If False they are not read,
            otherwise, this is a list of the requested lines.
        read_spectra (bool/list)
            Flag for whether to read spectra.
        spectra (dict, array-like, float)
            The spectra array from the grid. This is an N-dimensional
            grid where N is the number of axes of the SPS grid. The final
            dimension is always wavelength.
        lines (array-like, float)
            The lines array from the grid. This is an N-dimensional grid where
            N is the number of axes of the SPS grid. The final dimension is
            always wavelength.
        parameters (dict)
            A dictionary containing the grid's parameters used in its
            generation.
        axes (list, str)
            A list of the names of the spectral grid axes.
        naxes
            The number of axes the spectral grid has.
        logQ10 (dict)
            A dictionary of ionisation Q parameters.
        <grid_axis> (array-like, float)
            A Grid will always contain 1D arrays corresponding to the axes
            of the spectral grid. These are read dynamically from the HDF5
            file so can be anything but usually contain at least stellar ages
            and stellar metallicity.
        lam (array_like, float)
            The wavelengths at which the spectra are defined.
    """

    # Define Quantities
    lam = Quantity()

    def __init__(
        self,
        grid_name,
        grid_dir=None,
        read_spectra=True,
        read_lines=True,
        new_lam=None,
        filters=None,
    ):
        """
        Initailise the grid object, open the grid file and extracting the
        relevant data.

        Args:
            grid_name (str)
                The file name of the grid (if no extension is provided then
                hdf5 is assumed).
            grid_dir (str)
                The file path to the directory containing the grid file.
            read_spectra (bool)
                Should we read the spectra?
            read_lines (bool)
                Should we read the lines?
            new_lam (array-like, float)
                An optional user defined wavelength array the spectra will be
                interpolated onto, see Grid.interp_spectra.
            filters (FilterCollection)
                An optional FilterCollection object to unify the grids
                wavelength grid with. If provided, this will override new_lam
                whether passed or not.
        """
        if grid_dir is None:
            grid_dir = os.path.join(os.path.dirname(filepath), "data/grids")

        # The grid directory
        self.grid_dir = grid_dir

        # Have we been passed an extension?
        if (
            grid_name.split(".")[-1] == "hdf5"
            or grid_name.split(".")[-1] == "h5"
        ):
            self.grid_ext = grid_name.split(".")[-1]
        else:
            self.grid_ext = "hdf5"

        # Strip the extension off the name (harmless if no extension)
        self.grid_name = grid_name.replace(".hdf5", "").replace(".h5", "")

        # Construct the full path
        self.grid_filename = (
            f"{self.grid_dir}/{self.grid_name}.{self.grid_ext}"
        )

        # Flags for reading behaviour
        self.read_lines = read_lines
        self.read_spectra = read_spectra  # not used

        # Set up attributes we will set later
        self.spectra = None
        self.lines = None

        # Convert line list into a flattened list and remove duplicates
        if isinstance(read_lines, list):
            read_lines = flatten_linelist(read_lines)

        # Get basic info of the grid
        with h5py.File(self.grid_filename, "r") as hf:
            self.parameters = {k: v for k, v in hf.attrs.items()}

            # Get list of axes
            self.axes = list(hf.attrs["axes"])

            # Put the values of each axis in a dictionary
            self.axes_values = {
                axis: hf["axes"][axis][:] for axis in self.axes
            }

            # Set the values of each axis as an attribute
            # e.g. self.log10age == self.axes_values['log10age']
            for axis in self.axes:
                setattr(self, axis, self.axes_values[axis])

            # Number of axes
            self.naxes = len(self.axes)

            # If specific ionising photon luminosity is available set this
            # as an attribute as well
            if "log10_specific_ionising_luminosity" in hf.keys():
                self.log10_specific_ionising_lum = {}
                for ion in hf["log10_specific_ionising_luminosity"].keys():
                    self.log10_specific_ionising_lum[ion] = hf[
                        "log10_specific_ionising_luminosity"
                    ][ion][:]

            # If log10Q is available set this as an attribute as well
            # TODO: This needs to die
            if "log10Q" in hf.keys():
                self.log10Q = {}
                for ion in hf["log10Q"].keys():
                    self.log10Q[ion] = hf["log10Q"][ion][:]

        # Read in spectra
        if read_spectra:
            self.spectra = {}

            with h5py.File(
                f"{self.grid_dir}/{self.grid_name}.hdf5", "r"
            ) as hf:
                # Get list of available spectra
                spectra_ids = list(hf["spectra"].keys())

                # Remove wavelength dataset
                spectra_ids.remove("wavelength")

                # Remove normalisation dataset
                if "normalisation" in spectra_ids:
                    spectra_ids.remove("normalisation")

                for spectra_id in spectra_ids:
                    self.lam = hf["spectra/wavelength"][:]
                    self.spectra[spectra_id] = hf["spectra"][spectra_id][:]

            # If a full cloudy grid is available calculate some
            # other spectra for convenience.
            if "linecont" in self.spectra.keys():
                self.spectra["total"] = (
                    self.spectra["transmitted"] + self.spectra["nebular"]
                )

                self.spectra["nebular_continuum"] = (
                    self.spectra["nebular"] - self.spectra["linecont"]
                )

            # Save list of available spectra
            self.available_spectra = list(self.spectra.keys())

        if read_lines is not False:
            if check_lines_available(self.grid_name, self.grid_dir):
                pass
            else:
                raise Exception(
                    (
                        "No lines available on this grid object. "
                        "Either set `read_lines=False`, or load a grid "
                        "containing line information"
                    )
                )

            # If read_lines is True read all available lines in the grid,
            # otherwise if read_lines is a list just read the lines
            # in the list.

            self.lines = {}

            # If a list of lines is provided then only read lines in this list
            if isinstance(read_lines, list):
                read_lines = flatten_linelist(read_lines)

            # If a list isn't provided then use all available lines to the grid
            else:
                read_lines = get_available_lines(self.grid_name, self.grid_dir)

            with h5py.File(
                f"{self.grid_dir}/{self.grid_name}.hdf5", "r"
            ) as hf:
                for line in read_lines:
                    self.lines[line] = {}
                    self.lines[line]["wavelength"] = hf["lines"][line].attrs[
                        "wavelength"
                    ]
                    self.lines[line]["luminosity"] = hf["lines"][line][
                        "luminosity"
                    ][:]
                    self.lines[line]["continuum"] = hf["lines"][line][
                        "continuum"
                    ][:]

            # Save list of available lines
            self.available_lines = list(self.lines.keys())

        # Has a new wavelength grid been passed to interpolate
        # the spectra onto?
        if new_lam is not None and filters is None:
            # Double check we aren't being asked to do something impossible.
            if not read_spectra:
                raise exceptions.InconsistentArguments(
                    "Can't interpolate spectra onto a new wavelength array if"
                    " no spectra have been read in! Set read_spectra=True."
                )

            # Interpolate the spectra grid
            self.interp_spectra(new_lam)

        # Are we unifying with a filter collection?
        if filters is not None:
            # Double check we aren't being asked to do something impossible.
            if not read_spectra:
                raise exceptions.InconsistentArguments(
                    "Can't interpolate spectra onto a FilterCollection "
                    "wavelength array if no spectra have been read in! "
                    "Set read_spectra=True."
                )

            # Warn the user the new_lam will be ignored
            if new_lam is not None:
                print(
                    "If a FilterCollection is defined alongside new_lam "
                    "then FilterCollection.lam takes precedence and new_lam "
                    "is ignored"
                )

            self.unify_with_filters(filters)

    def interp_spectra(self, new_lam, loop_grid=False):
        """
        Interpolates the spectra grid onto the provided wavelength grid.

        NOTE: this will overwrite self.lam and self.spectra, overwriting
        the attributes loaded from the grid file. To get these back a new grid
        will need to instantiated with no lam argument passed.

        Args:
            new_lam (unyt_array/array-like, float)
                The new wavelength array to interpolate the spectra onto.
            loop_grid (bool)
                flag for whether to do the interpolation over the whole
                grid, or loop over the first axes. The latter is less memory
                intensive, but slower. Defaults to False.
        """

        # Handle and remove the units from the passed wavelengths if needed
        if isinstance(new_lam, unyt_array):
            if new_lam.units != self.lam.units:
                new_lam = new_lam.to(self.lam.units)
            new_lam = new_lam.value

        # Loop over spectra to interpolate
        for spectra_type in self.available_spectra:
            # Are we doing the look up in one go, or looping?
            if loop_grid:
                new_spectra = [None] * len(self.spectra[spectra_type])

                # Loop over first axis of spectra array
                for i, _spec in enumerate(self.spectra[spectra_type]):
                    new_spectra[i] = spectres(new_lam, self._lam, _spec)

                del self.spectra[spectra_type]
                new_spectra = np.asarray(new_spectra)
            else:
                # Evaluate the function at the desired wavelengths
                new_spectra = spectres(
                    new_lam, self._lam, self.spectra[spectra_type]
                )

            # Update this spectra
            self.spectra[spectra_type] = new_spectra

        # Update wavelength array
        self.lam = new_lam

    def __str__(self):
        """
        Function to print a basic summary of the Grid object.
        """

        # Set up the string for printing
        pstr = ""

        # Add the content of the summary to the string to be printed
        pstr += "-" * 30 + "\n"
        pstr += "SUMMARY OF GRID" + "\n"
        for axis in self.axes:
            pstr += f"{axis}: {getattr(self, axis)} \n"
        for k, v in self.parameters.items():
            pstr += f"{k}: {v} \n"
        if self.lines:
            pstr += f"available lines: {self.available_lines}\n"
        if self.spectra:
            pstr += f"available spectra: {self.available_spectra}\n"
        pstr += "-" * 30 + "\n"

        return pstr

    def get_nearest_index(self, value, array):
        """
        Function for calculating the closest index in an array for a
        given value.

        TODO: This could be moved to utils?

        Args:
            value (float/unyt_quantity)
                The target value.

            array (np.ndarray/unyt_array)
                The array to search.

        Returns:
            int
                The index of the closet point in the grid (array)
        """

        # Handle units on both value and array
        # First do we need a conversion?
        if isinstance(array, unyt_array) and isinstance(value, unyt_quantity):
            if array.units != value.units:
                value = value.to(array.units)

        # Get the values
        if isinstance(array, unyt_array):
            array = array.value
        if isinstance(value, unyt_quantity):
            value = value.value

        return (np.abs(array - value)).argmin()

    def get_grid_point(self, values):
        """
        Function to identify the nearest grid point for a tuple of values.

        Args:
            values (tuple)
                The values for which we want the grid point. These have to be
                in the same order as the axes.

        Returns:
            (tuple)
                A tuple of integers specifying the closest grid point.
        """

        return tuple(
            [
                self.get_nearest_index(value, getattr(self, axis))
                for axis, value in zip(self.axes, values)
            ]
        )

    def get_spectra(self, grid_point, spectra_id="incident"):
        """
        Function for creating an Sed object for a specific grid point.

        Args:
            grid_point (tuple)
                A tuple of integers specifying the closest grid point.
            spectra_id (str)
                The name of the spectra (in the grid) that is desired.

        Returns:
            sed (synthesizer.sed.Sed)
                A synthesizer Sed object
        """

        # Throw exception if the line_id not in list of available lines
        if spectra_id not in self.available_spectra:
            raise exceptions.InconsistentParameter(
                "Provided spectra_id is not in" "list of available spectra."
            )

        # Throw exception if the grid_point has a different shape from the grid
        if len(grid_point) != self.naxes:
            raise exceptions.InconsistentParameter(
                "The grid_point tuple provided"
                "as an argument should have same shape as the grid."
            )

        # TODO: throw an exception if grid point is outside grid bounds

        return Sed(self.lam, lnu=self.spectra[spectra_id][grid_point])

    def get_line(self, grid_point, line_id):
        """
        Function for creating a Line object for a given line_id and grid_point.

        Args:
            grid_point (tuple)
                A tuple of integers specifying the closest grid point.
            line_id (str)
                The id of the line.

        Returns:
            line (synthesizer.line.Line)
                A synthesizer Line object.
        """

        # Throw exception if the grid_point has a different shape from the grid
        if len(grid_point) != self.naxes:
            raise exceptions.InconsistentParameter(
                "The grid_point tuple provided"
                "as an argument should have same shape as the grid."
            )

        if type(line_id) is str:
            line_id = [line_id]

        wavelength = []
        luminosity = []
        continuum = []

        for line_id_ in line_id:
            # Throw exception if tline_id not in list of available lines
            if line_id_ not in self.available_lines:
                raise exceptions.InconsistentParameter(
                    "Provided line_id is" "not in list of available lines."
                )

            line_ = self.lines[line_id_]
            wavelength.append(line_["wavelength"])
            luminosity.append(line_["luminosity"][grid_point])
            continuum.append(line_["continuum"][grid_point])

        return Line(line_id, wavelength, luminosity, continuum)

    def get_lines(self, grid_point, line_ids=None):
        """
        Function a LineCollection of multiple lines.

        Args:
            grid_point (tuple)
                A tuple of the grid point indices.
            line_ids (list)
                A list of lines, if None use all available lines.

        Returns:
            lines (lines.LineCollection)
        """

        # If no line ids are provided calculate all lines
        if line_ids is None:
            line_ids = self.available_lines

        # Line dictionary
        lines = {}

        for line_id in line_ids:
            line = self.get_line(grid_point, line_id)

            # Add to dictionary
            lines[line.id] = line

        # Create and return collection
        return LineCollection(lines)

    def plot_specific_ionising_lum(
        self,
        ion="HI",
        hsize=3.5,
        vsize=None,
        cmap=cmr.sapphire,
        vmin=None,
        vmax=None,
        max_log10age=None,
    ):
        """
        Make a simple plot of the specific ionsing photon luminosity (logged)
        as a function of log10age and metallicity for a given grid and ion.

        Args:
           ion (str)
                The desired ion, most grids only have HI and HII calculated by
                default at present.

            hsize (float)
                The horizontal size of the figure

            vsize (float)
                The vertical size of the figure

            cmap (object/str)
                A colourmap object or string defining the matplotlib colormap.

            vmin (float)
                The minimum specific ionising luminosity used in the colourmap

            vmax (float)
                The maximum specific ionising luminosity used in the colourmap

            max_log10age (float)
                The maximum log10(age) to plot

        Returns:
            matplotlib.Figure
                The created figure containing the axes.
            matplotlib.Axis
                The axis on which to plot.
        """

        # Define the axis coordinates
        left = 0.2
        height = 0.65
        bottom = 0.15
        width = 0.75

        # Scale the plot height if necessary
        if vsize is None:
            vsize = hsize * width / height

        # Create the figure
        fig = plt.figure(figsize=(hsize, vsize))

        # Create the axes
        ax = fig.add_axes((left, bottom, width, height))
        cax = fig.add_axes([left, bottom + height + 0.01, width, 0.05])

        # Create an index array
        y = np.arange(len(self.metallicity))

        # Select grid for specific ion
        if hasattr(self, "log10_specific_ionising_lum"):
            log10_specific_ionising_lum = self.log10_specific_ionising_lum[ion]
        else:
            log10_specific_ionising_lum = self.log10Q[ion]

        # Truncate grid if max age provided
        if max_log10age is not None:
            ia_max = self.get_nearest_index(max_log10age, self.log10age)
            log10_specific_ionising_lum = log10_specific_ionising_lum[
                :ia_max, :
            ]
        else:
            ia_max = -1

        # If no limits are supplied set a sensible range for HI ion otherwise
        # use min max
        if ion == "HI":
            if vmin is None:
                vmin = 42.5
            if vmax is None:
                vmax = 47.5
        else:
            if vmin is None:
                vmin = np.min(log10_specific_ionising_lum)
            if vmax is None:
                vmax = np.max(log10_specific_ionising_lum)

        # Plot the grid of log10_specific_ionising_lum
        ax.imshow(
            log10_specific_ionising_lum.T,
            origin="lower",
            extent=[
                self.log10age[0],
                self.log10age[ia_max],
                y[0] - 0.5,
                y[-1] + 0.5,
            ],
            cmap=cmap,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )

        # Define the normalisation for the colorbar
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        cmapper.set_array([])

        # Add colourbar
        fig.colorbar(cmapper, cax=cax, orientation="horizontal")
        cax.xaxis.tick_top()
        cax.xaxis.set_label_position("top")
        cax.set_xlabel(
            r"$\rm log_{10}(\dot{n}_{" + ion + "}/s^{-1}\\ M_{\\odot}^{-1})$"
        )
        cax.set_yticks([])

        # Set custom tick marks
        ax.set_yticks(y, self.metallicity)
        ax.minorticks_off()

        # Set labels
        ax.set_xlabel("$\\log_{10}(\\mathrm{age}/\\mathrm{yr})$")
        ax.set_ylabel("$Z$")

        return fig, ax

    def get_delta_lambda(self, spectra_id="incident"):
        """
        Calculates the delta lambda for the given spectra.

        Args:
            spectra_id (str)
                Identifier for the spectra (default is "incident").

        Returns:
            tuple
                A tuple containing the list of wavelengths and delta lambda.
        """

        # Calculate delta lambda for each wavelength
        delta_lambda = np.log10(self.lam[1:]) - np.log10(self.lam[:-1])

        # Return tuple of wavelengths and delta lambda
        return self.lam, delta_lambda

    def get_sed(self, spectra_type):
        """
        Get the spectra grid as an Sed object enabling grid wide
        use of Sed methods for flux, photometry, indices, ionising photons
        etc.

        Args:
            spectra_type (string)
                The key of the spectra grid to extract as an Sed object.

        Returns:
            Sed
                The spectra grid as an Sed object.
        """
        return Sed(self.lam, self.spectra[spectra_type])

    def unify_with_filters(self, filters):
        """
        Unify the grid with a FilterCollection object.

        This will:
        - Find the Grid wavelengths at which transmission is non-zero.
        - Limit the Grid's spectra and wavelength array to where transmision
          is non-zero.
        - Interpolate the filter collection onto the Grid's new wavelength
          array.

        Args:
            filters (synthesizer.filter.FilterCollection)
                The FilterCollection object to unify with this grid.
        """
        # Get the minimum and maximum wavelengths with non-zero transmission
        min_lam, max_lam = filters.get_non_zero_lam_lims()

        # Define the mask to eliminate wavelengths outside this range
        okinds = np.logical_and(self.lam >= min_lam, self.lam <= max_lam)

        # Apply the mask to the grid wavelengths
        self.lam = self.lam[okinds]

        # Apply the mask to the spectra
        for spectra_type in self.available_spectra:
            self.spectra[spectra_type] = self.spectra[spectra_type][
                ..., okinds
            ]

        # Interpolate the filters onto this new wavelength range
        filters.resample_filters(new_lam=self.lam)

    def animate_grid(
        self,
        show=False,
        save_path=None,
        fps=30,
        spectra_type="incident",
    ):
        """
        Create an animation of the grid stepping through wavelength.

        Each frame of the animation is a wavelength bin.

        Args:
            show (bool):
                Should the animation be shown?
            save_path (str, optional):
                Path to save the animation. If not specified, the
                animation is not saved.
            fps (int, optional):
                the number of frames per second in the output animation.
                Default is 30 frames per second.

        Returns:
            matplotlib.animation.FuncAnimation: The animation object.
        """
        # Create the figure and axes
        fig, ax = plt.subplots(1, 1, figsize=(6, 8))

        # Get the spectra grid
        spectra = self.spectra[spectra_type]

        # Get the normalisation
        vmin = 10**10
        vmax = np.percentile(spectra, 99.9)

        # Define the norm
        norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)

        # Create a placeholder image
        img = ax.imshow(
            spectra[:, :, 0],
            extent=[
                self.log10age.min(),
                self.log10age.max(),
                self.metallicity.min(),
                self.metallicity.max(),
            ],
            origin="lower",
            animated=True,
            norm=norm,
            aspect="auto",
        )

        cbar = fig.colorbar(img)
        cbar.set_label(
            r"$L_{\nu}/[\mathrm{erg s}^{-1}\mathrm{ Hz}^{-1} "
            r"\mathrm{ M_\odot}^{-1}]$"
        )

        ax.set_title(f"Wavelength: {self.lam[0]:.2f}{self.lam.units}")
        ax.set_xlabel("$\\log_{10}(\\mathrm{age}/\\mathrm{yr})$")
        ax.set_ylabel("$Z$")

        def update(i):
            # Update the image for the ith frame
            img.set_data(spectra[:, :, i])
            ax.set_title(f"Wavelength: {self.lam[i]:.2f}{self.lam.units}")
            return [
                img,
            ]

        # Calculate interval in milliseconds based on fps
        interval = 1000 / fps

        # Create the animation
        anim = FuncAnimation(
            fig, update, frames=self.lam.size, interval=interval, blit=False
        )

        # Save if a path is provided
        if save_path is not None:
            anim.save(save_path, writer="imagemagick")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return anim
