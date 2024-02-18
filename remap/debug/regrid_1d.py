from astropy.io import ascii
from astropy.table import Table
import numpy
import matplotlib.pyplot as plt

# Load in Bin 2 model
bin2_model = Table.read(
    "../../grids-new/Py.Bin2.txt", format="ascii.fixed_width_two_line"
)
print(bin2_model)

# Find r_min and r_max.
r_min = bin2_model["r"].min()
r_max = bin2_model["r"][:-2].max()
print(f"{r_min = :e} {r_max = :e}")

# Create three grids of varying resolution by interpolating on the grid

table_columns = {
    "i": numpy.int32,
    "r": numpy.float32,
    "v_r": numpy.float32,
    "rho": numpy.float32,
    "t_r": numpy.float32,
    "t_e": numpy.float32,
}

bin2_models = bin2_model[1:-2]

for n_cells in [15, 100, 500, 5000, 10000]:
    r_points = numpy.logspace(numpy.log10(r_min), numpy.log10(r_max), n_cells + 3)
    new_grid = Table(
        # data=None, names=table_columns.keys(), dtype=table_columns.values()
    )

    new_grid["i"] = numpy.arange(0, n_cells + 3, 1)
    new_grid["r"] = r_points
    new_grid["v_r"] = numpy.interp(r_points, bin2_model["r"], bin2_model["v_r"])
    new_grid["rho"] = numpy.interp(r_points, bin2_model["r"], bin2_model["rho"])
    new_grid["t_r"] = numpy.interp(r_points, bin2_model["r"], bin2_model["t_e"])
    new_grid["t_e"] = numpy.interp(r_points, bin2_model["r"], bin2_model["t_r"])

    new_grid["rho"][0] = new_grid["t_e"][0] = new_grid["t_r"][0] = 0
    new_grid["rho"][-1] = new_grid["t_e"][-1] = new_grid["t_r"][-1] = 0
    new_grid["rho"][-2] = new_grid["t_e"][-2] = new_grid["t_r"][-2] = 0

    print(new_grid)

    ascii.write(
        new_grid,
        f"Py.Bin2.{n_cells}.txt",
        format="fixed_width_two_line",
        overwrite=True,
    )

    plt.loglog(new_grid["r"], new_grid["v_r"], label="v_r")
    plt.xlabel("r")
    plt.show()

    plt.loglog(new_grid["r"], new_grid["rho"], label="rho")
    plt.xlabel("r")
    plt.show()
