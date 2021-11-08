import create_2d_grid
import numpy as np
import pypython
from astropy.table import Table


create_2d_grid.OUTPUT_RADIAL_VELOCTY = True
create_2d_grid.OUTPUT_TEMPERATURE = True
create_2d_grid.INTERPOLATION_LEVEL = "cubic"
original, regrid = create_2d_grid.regrid_model(1.23391, 1000)
grid = create_2d_grid.setup_grid_boundaries(regrid)


def stefan_boltzmann(t, r_in):
    sb = pypython.constants.STEFAN_BOLTZMANN
    return 4 * np.pi * r_in**2 * sb * t**4

# N. Roth models

the_1d_models = [
    [67.5, 87.4, 3.221062e+12, 4.0e47, "Dai.Bin1"],
    # [67.5, 87.4, 2.500000e+13, 4.0e47, "Dai.Bin1"],  # tau_es < 100
    [45.0, 67.5, 3.549527e+12, 3.0e46, "Dai.Bin2"],
    [22.5, 45.0, 4.310358e+12, 3.0e46, "Dai.Bin3"],
    [5.70, 22.5, 7.004391e+12, 3.0e46, "Dai.Bin4"]
]

# Python comparisons

rg_bc = pypython.physics.blackhole.gravitational_radius(5e6) * 30

the_1d_models = [
    [67.5, 87.4, rg_bc, 4.0e47, "Py.Bin1"],
    [45.0, 67.5, rg_bc, 3.0e46, "Py.Bin2"],
    [22.5, 45.0, rg_bc, 3.0e46, "Py.Bin3"],
    [5.70, 22.5, rg_bc, 3.0e46, "Py.Bin4"]
]


nx, nz = np.max(grid["i"]) + 1, np.max(grid["j"]) + 1

theta_points = grid["theta"][:nz - 1]
r_points = grid["r"][::nz]

for theta1, theta2, r_in, l_in, name in the_1d_models:

    print(name)
    
    theta_idx1 = pypython.get_array_index(theta_points, theta1)
    theta_idx2 = pypython.get_array_index(theta_points, theta2)

    r = []
    v_r = []
    rho = []
    t_r = []

    rho = np.zeros_like(r_points)
    v_r = np.zeros_like(r_points)
    t_r = np.zeros_like(r_points)

    for n in range(1, nx - 1):  # loop over each radius
        r_idx = n * nz
        r_min = grid["r"][r_idx]
        r_max = grid["r"][((n + 1) * nz)]

        if r_min < r_in:
            continue

        total_volume = 0

        for i in range(r_idx + theta_idx1, r_idx + theta_idx2): # loop over the theta cells

            theta_min = np.deg2rad(grid["theta"][i])
            theta_max = np.deg2rad(grid["theta"][i + 1])

            cell_volume = (2.0 / 3.0) * np.pi * (r_max**3 - r_min**3) * (np.cos(theta_min) - np.cos(theta_max)) * 2 * np.pi
            total_volume += cell_volume

            if grid["v_r"][i] < 0:
                grid["v_r"][i] = 0

            rho[n] += grid["rho"][i] * cell_volume
            v_r[n] += grid["v_r"][i] * cell_volume
            t_r[n] += 10**grid["t_r"][i] * cell_volume

        rho[n] /= total_volume 
        v_r[n] /= total_volume
        t_r[n] /= total_volume

    first_non_zero = np.nonzero(rho)[0][0]

    table = Table()
    table["i"] = np.arange(0, len(r_points[first_non_zero:]), 1)
    table["r"] = r_points[first_non_zero:]
    table["v_r"] = v_r[first_non_zero:]
    table["rho"] = rho[first_non_zero:]
    table["t_e"] = 0.9 * t_r[first_non_zero:]
    table["t_r"] = t_r[first_non_zero:]

    table["r"].format = "%1.6e"
    table["v_r"].format = "%1.6e"
    table["rho"].format = "%1.6e"
    table["t_e"].format = "%1.6e"
    table["t_r"].format = "%1.6e"

    for i in range(1):
        table[-(i + 1)]["v_r"] = table[-2]["v_r"]

    for i in range(3):
        table[-i]["rho"] = 0
        table[-i]["t_e"] = 0
        table[-i]["t_r"] = 0

    table.write(f"{name}.txt", format="ascii.fixed_width_two_line", overwrite=True)
