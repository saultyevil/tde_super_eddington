#!/usr/bin/env python3

"""The purpose of this script is to take in Jane Dai's non-linear grid
where ALL quantities are defined at cell center and to re-grid it onto a
logarithmic polar grid which Python can understand. This will ensure
that the grid coordinates are at the vertices of the cell, whilst the
density and etc are at the center.
"""

import argparse
from sys import exit

import numpy as np
from astropy.io import ascii
from astropy.table import Table
from pypython import constants
from scipy.interpolate import griddata

from plotting import plot_regrid

VERBOSE = False
SHOW_PLOTS = False
OUTPUT_TEMPERATURE = True
OUTPUT_RADIAL_VELOCTY = False
VY_KEPLERIAN = False

INWIND = 0
NOT_INWIND = -1

INTERPOLATION_LEVEL = "cubic"
RG_MIN = 30


def setup_script():
    """Parse command line arguments to setup the script."""

    p = argparse.ArgumentParser()
    p.add_argument(
        "--keplerian",
        action="store_true",
        help="Use a Keplerian velocity profile for the y direction",
    )
    p.add_argument(
        "--radial", action="store_true", help="Output radial velocity as well"
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Show plots of the regridded and original model",
    )
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = p.parse_args()

    if args.keplerian:
        global VY_KEPLERIAN
        VY_KEPLERIAN = True
    if args.radial:
        global OUTPUT_RADIAL_VELOCTY
        OUTPUT_RADIAL_VELOCTY = True
    if args.plot:
        global SHOW_PLOTS
        SHOW_PLOTS = True
    if args.verbose:
        global VERBOSE
        VERBOSE = True


def read_original_model():
    """Read in the original model, assuming that it has been converted into
    astropy's ascii.fixed_wdith_two_line format.
    """

    mbh = 5e6 * constants.MSOL
    rg = constants.G * mbh / constants.C**2

    return Table.read("../model.txt", format="ascii.fixed_width_two_line"), mbh, rg


def interpolate_points(variable, variable_name, old_r, old_theta, new_r, new_theta):
    """Interpolate the old grid onto the new grid."""

    grid = griddata(
        (old_r, old_theta),
        variable,
        (new_r, new_theta),
        method=INTERPOLATION_LEVEL,
        fill_value=-1,
    )

    n_error = 0
    for i, element in enumerate(grid):
        if element == -1:
            n_error += 1

            grid[i] = griddata(
                (old_r, old_theta), variable, (new_r[i], new_theta[i]), method="nearest"
            )

            if VERBOSE:
                print(
                    f"{variable_name} interpolation problem for at r = {new_r[i]} theta = {np.rad2deg(new_theta[i])}"
                )

    if n_error:
        print(f"There were {n_error} interpolation errors for {variable_name}")

    return grid


def regrid_model(rg_cutoff: float = None, rmax: float = None):
    """Regrid the original nolinear grid onto a logarithmic grid which will
    follow's Python's conventions.
    Grid spacings are calculated using logarithmic coordinates for the r axis,
    but linear spacings are used for the theta axis (as standard in Python).
    The r and theta units are in units of Rg and radians respectively. The -2
    and -3 when calculating the spacing on either axis is due to the number
    of ghosts cells on a Python grid.

    Parameters
    ----------
    rg_cutoff: float [optional]
        Any cell contained within a sphere of radius rg_cutoff will be thrown
        away.
    """

    # Read in the original grid, get the r and theta coordinates

    original_model, mbh, rg = read_original_model()

    # Create a table to hold the re-gridded data in Python's format

    rg_mdim = 192  # r
    rg_ndim = 64  # theta

    if rg_cutoff is None:
        rg_rmin = np.min(original_model["r"])
    else:
        rg_rmin = rg_cutoff

    if rmax is not None:
        rg_rmax = rmax
    else:
        rg_rmax = np.max(original_model["r"])

    print(f"Rg min {rg_rmin} ({rg_rmin* rg:1.6e} cm)")
    print(f"Rg max    {rg_rmax} ({rg_rmax * rg:1.6e} cm)")

    columns = {
        "i": np.int64,
        "j": np.int64,
        "inwind": np.int64,
        "r": np.float64,
        "theta": np.float64,
        "v_x": np.float64,
        "v_y": np.float64,
        "v_z": np.float64,
        "rho": np.float64,
    }

    if OUTPUT_TEMPERATURE:
        columns["t_e"] = np.float64
        columns["t_r"] = np.float64

    if OUTPUT_RADIAL_VELOCTY:
        columns["v_r"] = np.float64
        columns["v_theta"] = np.float64

    regridded_model = Table()  # names=columns.keys(), dtype=columns.values())

    # Interpolate the velocity here because vectorised code is faster :-)

    dtheta = 90 / (rg_ndim - 2)
    log_dr = np.log10(rg_rmax / rg_rmin) / (rg_mdim - 3)

    r_points = []
    r_cen_points = []
    theta_points = []
    theta_cen_points = []
    i_index = []
    j_index = []

    # Create arrays of the grid points, I'm doing it manually here rather
    # than using numpy, because I don't know. I think it's more explicit what
    # is happening for when I come back to try and compare this to what is
    # going on in Python
    # The coordinates generated here are assumed to be the lower (left) vertex
    # of the cell, as Python requires.

    nelem_current = 0

    for i in range(rg_mdim):
        for j in range(rg_ndim):
            nelem_current += 1

            theta = j * dtheta
            theta_cen = theta + 0.5 * dtheta

            # Python, the program, puts a limit of 90 degrees. I don't like how,
            # we do this, but I do not see Python being redesigned anytime soon
            # so we must do it like that to ensure compatability

            if theta > 90:
                theta = 90
            if theta_cen > 90:
                theta_cen = 90

            r = rg_rmin * 10 ** (log_dr * (i - 1))
            r_cen = 0.5 * rg_rmin * (10 ** (log_dr * i) + 10 ** (log_dr * (i - 1)))

            r_points.append(r)
            r_cen_points.append(r_cen)
            theta_points.append(theta)
            theta_cen_points.append(theta_cen)
            i_index.append(i)
            j_index.append(j)

    r_points = np.array(r_points, dtype=np.float64)
    r_cen_points = np.array(r_cen_points, dtype=np.float64)

    # convert to radians because the smaller numbers work better with the interpolation
    # function we use
    theta_points = np.deg2rad(np.array(theta_points, dtype=np.float64))
    theta_cen_points = np.deg2rad(np.array(theta_cen_points, dtype=np.float64))

    i_index = np.array(i_index, dtype=np.int64)
    j_index = np.array(j_index, dtype=np.int64)

    # Interpolate the cell centered velocties and densities to the cell
    # vertices. By default, we use linear or cubic interpolation. This will
    # probably be good enough. In the case the interpolation fails, we use
    # a nearest neighbour interpoloation scheme, which isn't great but it will
    # fill in the blanks.

    if INTERPOLATION_LEVEL not in ["nearest", "linear", "cubic"]:
        print(f"Invalid value for INTERPOLATION_LEVEL {INTERPOLATION_LEVEL}")
        exit(1)

    # These are defined at the cell inner vertex

    v_r = interpolate_points(
        original_model["v_r"],
        "v_r",
        original_model["r"],
        original_model["theta"],
        r_points,
        theta_points,
    )
    v_theta = interpolate_points(
        original_model["v_theta"],
        "v_theta",
        original_model["r"],
        original_model["theta"],
        r_points,
        theta_points,
    )

    # These are cell centred quantities

    rho = interpolate_points(
        original_model["rho"],
        "rho",
        original_model["r"],
        original_model["theta"],
        r_cen_points,
        theta_cen_points,
    )

    if OUTPUT_TEMPERATURE:
        t_r = interpolate_points(
            original_model["t_rad"],
            "t_rad",
            original_model["r"],
            original_model["theta"],
            r_cen_points,
            theta_cen_points,
        )

    # For vy, use a Keplerian velocity. Jane said this should be OK.

    v_r *= constants.VLIGHT
    v_theta *= constants.VLIGHT

    vx = v_r * np.sin(theta_points) + v_theta * np.cos(theta_points)
    vz = v_r * np.cos(theta_points) - v_theta * np.sin(theta_points)

    # vy is either Keplerian, or just set to 0

    if VY_KEPLERIAN:
        vy = np.sqrt(constants.G * mbh / (r_points * rg))
    else:
        vy = np.zeros_like(vx)

    # Add the data to the table

    theta_points = np.rad2deg(theta_points)

    regridded_model["i"] = i_index
    regridded_model["j"] = j_index
    regridded_model["inwind"] = np.ones_like(i_index, dtype=int) * int(INWIND)
    regridded_model["r"] = np.array(r_points) * rg
    regridded_model["theta"] = theta_points
    regridded_model["v_x"] = vx
    regridded_model["v_y"] = vy
    regridded_model["v_z"] = vz
    regridded_model["rho"] = 10 ** np.array(rho)
    regridded_model["t_e"] = 10 ** (0.9 * np.array(t_r))
    regridded_model["t_r"] = 10**t_r

    if OUTPUT_RADIAL_VELOCTY:
        regridded_model["v_r"] = v_r
        regridded_model["v_theta"] = v_theta

    for column in list(columns.keys())[3:]:
        regridded_model[column].format = "%1.6e"

    first_shell = regridded_model[regridded_model["i"] == 1]
    mean_t_r = np.median(first_shell["t_r"])
    print(f"Median T_r at inner radius = {mean_t_r:1.6e}")

    return original_model, regridded_model


def setup_grid_boundaries(rt: Table):
    """Add the boundary cells to the grid. This means we have a layer of cells
    which are marked as not being in the wind. The density and temperature are
    set to 0, but they still have a velocity.
    Parameters
    ----------
    rt: astropy.table.Table
        The grid to add the boundary cells onto."""

    ndim = np.max(rt["j"]) + 1

    # The 2 outer layers

    for i in range(1, 2 * ndim + 1):
        rt[-i]["inwind"] = NOT_INWIND
        rt[-i]["rho"] = 0
        if OUTPUT_TEMPERATURE:
            rt[-i]["t_e"] = rt[-i]["t_r"] = 0

    # The inner layer

    for i in range(ndim):
        rt[i]["inwind"] = NOT_INWIND
        rt[i]["rho"] = 0
        if OUTPUT_TEMPERATURE:
            rt[i]["t_e"] = rt[i]["t_r"] = 0

    # pylint: disable=C0200
    for i in range(len(rt)):
        # Set anything where 5 < theta > 90 as being NOT_INWIND
        if rt[i]["theta"] < 5 or rt[i]["theta"] >= 90:
            rt[i]["inwind"] = NOT_INWIND
            rt[i]["rho"] = 0
            if OUTPUT_TEMPERATURE:
                rt[i]["t_e"] = rt[i]["t_r"] = 0

        # This is a bit of a hack to remove regions of the grid where v > c (usually
        # this is the jet region)
        # this should only be a problem if there are cells with r < 2 Rg
        v = np.sqrt(rt[i]["v_x"] ** 2 + rt[i]["v_y"] ** 2 + rt[i]["v_z"] ** 2)
        if v > constants.C:
            rt[i]["inwind"] = NOT_INWIND
            rt[i]["rho"] = 0
            if OUTPUT_TEMPERATURE:
                rt[i]["t_e"] = rt[i]["t_r"] = 0

    return rt


def show_plots(original_grid: Table, new_grid: Table):
    """Show plots of the regridded model against the original.
    Parameters
    ----------
    original_grid: astropy.table.Table
        The original grid.
    new_grid: astropy.table.Table
        The new re-gridded grid."""

    plot_regrid.plot_fast(new_grid)
    plot_regrid.plot_inwind(new_grid)

    plot_regrid.doit(original_grid, new_grid, "rho", "Mass density")

    if OUTPUT_TEMPERATURE:
        plot_regrid.doit(original_grid, new_grid, "t_r", "Radiation Temperature")

    if OUTPUT_RADIAL_VELOCTY:
        plot_regrid.doit(original_grid, new_grid, "v_r", "Radial velocity")
        plot_regrid.doit(original_grid, new_grid, "v_theta", "Theta velocity")


if __name__ == "__main__":
    setup_script()
    original_grid, new_grid = regrid_model(30)  # paper is 30 rg
    new_grid = setup_grid_boundaries(new_grid)

    if SHOW_PLOTS:
        show_plots(original_grid, new_grid)

    # Write the final grid to file and print and plot

    if not OUTPUT_RADIAL_VELOCTY:
        ascii.write(
            new_grid,
            "../grids-new/Py.2d_regrid.txt",
            format="fixed_width_two_line",
            overwrite=True,
        )
    else:
        ascii.write(
            new_grid,
            "../grids-new/Py.2d_regrid.radial.txt",
            format="fixed_width_two_line",
            overwrite=True,
        )
