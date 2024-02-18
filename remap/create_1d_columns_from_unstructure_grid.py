#!/usr/bin/env python3

"""
The hope of this script is to re-create the 1D models which Nathan Roth used
in Jane's paper to create spectra.
"""


import argparse

import numpy as np
from astropy.io import ascii
from astropy.table import Table
from pypython import constants
from scipy.interpolate import interp1d


VERBOSE = False
SHOW_PLOTS = False
OUTPUT_TEMPERATURE = True
INTERPOLATION_LEVEL = "cubic"


BINS = [
    [67.5, 87.4, "Col67-87", "Bin1", 2.500000e+13],
    [45, 67.5, "Col45-67", "Bin2", 3.549527e+12],
    [22.5, 45, "Col22-45", "Bin3", 4.310358e+12],
    [5.7, 22.5, "Col5-22", "Bin4", 7.004391e+12]
]


def setup_script():
    """Parse command line arguments to setup the script."""

    p = argparse.ArgumentParser()
    p.add_argument("--plot", action="store_true", help="Show plots of the regridded and original model")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = p.parse_args()

    if args.plot:
        global SHOW_PLOTS
        SHOW_PLOTS = True
    if args.verbose:
        global VERBOSE
        VERBOSE = True


def read_original_model():
    """Read in the original model, assuming that it has been converted into
    astropy's ascii.fixed_wdith_two_line format."""

    mbh = 5e6 * constants.MSOL
    rg = constants.G * mbh / constants.C ** 2

    return Table.read("../model.txt", format="ascii.fixed_width_two_line"), mbh, rg


def create_columns():
    """Create angle averaged columns."""

    t, mbh, rg = read_original_model()
    mdim = 128
    ndim = 64
    sph_mdim = 128

    orig_r_points = np.array(t["r"][::ndim], dtype=np.float64)
    rmin = np.min(t["r"])
    
    # rmax = np.max(t["r"])
    rmax = 1000

    columns = {
        "i": np.int64,
        "r": np.float64,
        "v_r": np.float64,
        # "v_theta": np.float64,
        "rho": np.float64
    }

    if OUTPUT_TEMPERATURE:
        columns["t_e"] = np.float64
        columns["t_r"] = np.float64


    # Create coordinates for the grid, do this manually rather than using
    # numpy to make it clearer that the gridding is correct with how Python
    # does it.

    r_points = np.zeros(sph_mdim)
    r_mid_points = np.zeros_like(r_points)
    dr = np.log10(rmax / rmin) / (sph_mdim - 3)
    
    for i in range(sph_mdim):
        r_points[i] = rmin * 10 ** (dr * (i - 1))
        r_mid_points[i] = 0.5 * rmin * (10 ** (dr * i) + 10 ** (dr * (i - 1)))

    # Now create the columns
    # todo: need to start at cells where v_theta < 0, see the
    # last paragraph in Section 2.2 in Dai+2018

    for bin in BINS:

        name = bin[3]
        theta1 = np.deg2rad(bin[0])
        theta2 = np.deg2rad(bin[1])
        r_in = bin[-1]

        rt = Table(names=columns.keys(), dtype=columns.values())

        if INTERPOLATION_LEVEL not in ["nearest", "linear", "quadratic", "cubic"]:
            print(f"Invalid value for INTERPOLATION_LEVEL {INTERPOLATION_LEVEL}")
            exit(1)

        for i in range(mdim):
            start = i * ndim
            end = (i + 1) * ndim
            wedge = t[start:end]

            # This bit here figures out the indices required to average over
            # to fit in the angle bin. Probably not the fastest way to do it,
            # but since the range is small then these while loops shouldn't
            # take up a lot of run time

            j = k = 0
            while wedge["theta"][j] < theta1:
                j += 1
            while wedge["theta"][k] < theta2:
                k += 1

            row = [
                i,                               # i index
                orig_r_points[i],                # r points on original grid (centre of cell)
                np.mean(wedge["v_r"][j:k]),      # angle averaged v_r on original grid (centre of cell)
                # np.mean(wedge["v_theta"][j:k]),  # v_theta, only held for special reasons later
                np.mean(wedge["rho"][j:k])       # angle averaged rho on origin grid (centre of cell)
            ]

            if OUTPUT_TEMPERATURE:
                row += [
                    0,                            # electron temperature -- set to 0 for now
                    np.mean(wedge["t_rad"][j:k])  # radiation temperature
                ]

            rt.add_row(row)

        # Create interpolation functions, as we want these values on our new
        # grid points, and defined at cell edges for v_r and cell centre for rho

        v_r_interp = interp1d(rt["r"], rt["v_r"], INTERPOLATION_LEVEL, fill_value="extrapolate")
        rho_interp = interp1d(rt["r"], rt["rho"], INTERPOLATION_LEVEL, fill_value="extrapolate")

        # Note how we are now putting r_points, i.e. our defined points, into
        # rt instead, which are the *cell edges*. We thus need to interpolate
        # rho and the temperatures to the cell centres

        rt["v_r"] = v_r_interp(r_points) * constants.C
        rt["rho"] = 10 ** rho_interp(r_mid_points)

        if OUTPUT_TEMPERATURE:
            t_rad_interp = interp1d(rt["r"], rt["t_r"], INTERPOLATION_LEVEL, fill_value="extrapolate")
            rt["t_r"] = 10 ** t_rad_interp(r_mid_points)
            rt["t_e"] = 0.9 * rt["t_r"]  # from Lucy t_rad = 1.1 t_e

        rt["r"] = r_points * rg

        # Now find there v_r goes negative, then remove all the cells before
        # that

        for where, r in enumerate(rt["r"]):
            if r > r_in:
                break

        where -= 1
        print(rt["r"][where]/rg, rt["r"][where])
        rt.remove_rows(slice(where))

        # Re-do the cell indices

        for i in range(len(rt)):
            rt["i"][i] = i

        # Set the density to 0 for ghost cells. Note that -0, i.e. the first
        # iteration, will set the 0th cell :-)

        for i in range(3):
            rt["rho"][-i] = 0
            if OUTPUT_TEMPERATURE:
                rt["t_e"][-i] = rt["t_r"][-i] = 0

        # This is as far as we need to go, actually. Python will understand
        # this format quite well

        for thing in ["r", "v_r", "rho", "t_e", "t_r"]:
            rt[thing].format = "%1.6e"

        print(rt["r"]/rg)

        ascii.write(rt, f"../grids/EP.Dai.{name}.txt", format="fixed_width_two_line", overwrite=True)


if __name__ == "__main__":
    setup_script()
    create_columns()
