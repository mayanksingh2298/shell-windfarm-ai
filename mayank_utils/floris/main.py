import os

import numpy as np
import matplotlib.pyplot as plt

import floris.tools as wfct
import floris.tools.visualization as vis
from floris.tools.optimization.scipy.layout import LayoutOptimization


# Instantiate the FLORIS object
# file_dir = os.path.dirname(os.path.abspath(__file__))
fi = wfct.floris_interface.FlorisInterface(
    # os.path.join(file_dir, "../../../example_input.json")
    "example_input.json"
)

# Set turbine locations to 3 turbines in a triangle
D = fi.floris.farm.turbines[0].rotor_diameter
# layout_x = [10, 10, 10 + 7 * D]
# layout_y = [200, 1000, 200]
layout_x = [333.00000000,
            333.00000000,
            50.00000000,
            3950.00000000,
            749.75000000,
            749.75000000,
            50.00000000,
            3950.00000000,
            1166.50000000,
            1166.50000000,
            50.00000000,
            3950.00000000,
            1583.25000000,
            1583.25000000,
            50.00000000,
            3950.00000000,
            2000.00000000,
            2000.00000000,
            50.00000000,
            3950.00000000,
            2416.75000000,
            2416.75000000,
            50.00000000,
            3950.00000000,
            2833.50000000,
            2833.50000000,
            50.00000000,
            3950.00000000,
            3250.25000000,
            3250.25000000,
            50.00000000,
            3950.00000000,
            3667.00000000,
            3667.00000000,
            50.00000000,
            3950.00000000,
            1051.00000000,
            1051.00000000,
            1051.00000000,
            1051.00000000,
            1683.66666667,
            1683.66666667,
            1683.66666667,
            1683.66666667,
            2316.33333333,
            2316.33333333,
            2316.33333333,
            2316.33333333,
            2949.00000000,
            2949.00000000]
layout_y = [
            50.00000000,
            3950.00000000,
            333.00000000,
            333.00000000,
            50.00000000,
            3950.00000000,
            749.75000000,
            749.75000000,
            50.00000000,
            3950.00000000,
            1166.50000000,
            1166.50000000,
            50.00000000,
            3950.00000000,
            1583.25000000,
            1583.25000000,
            50.00000000,
            3950.00000000,
            2000.00000000,
            2000.00000000,
            50.00000000,
            3950.00000000,
            2416.75000000,
            2416.75000000,
            50.00000000,
            3950.00000000,
            2833.50000000,
            2833.50000000,
            50.00000000,
            3950.00000000,
            3250.25000000,
            3250.25000000,
            50.00000000,
            3950.00000000,
            3667.00000000,
            3667.00000000,
            1051.00000000,
            1683.66666667,
            2316.33333333,
            2949.00000000,
            1051.00000000,
            1683.66666667,
            2316.33333333,
            2949.00000000,
            1051.00000000,
            1683.66666667,
            2316.33333333,
            2949.00000000,
            1051.00000000,
            1683.66666667
        ]
fi.reinitialize_flow_field(layout_array=(layout_x, layout_y))

# Define the boundary for the wind farm
boundaries = [[3950.0, 3950.0], [3950.0, 50.0], [50.0, 50.0], [50.0, 3950.0]]

# Generate random wind rose data
# wd = np.arange(0.0, 360.0, 5.0)
# np.random.seed(1)
# ws = 8.0 + np.random.randn(len(wd)) * 0.5
# freq = np.abs(np.sort(np.random.randn(len(wd))))
# freq = freq / freq.sum()
wd = np.array([10.,10.,10.])
ws = np.array([10.,15.,20.])
freq = np.array([0.2,0.6,0.2])


# import pdb
# pdb.set_trace()

# Set optimization options
opt_options = {"maxiter": 50, "disp": True, "iprint": 2, "ftol": 1e-8}

# Compute initial AEP for optimization normalization
AEP_initial = fi.get_farm_AEP(wd, ws, freq)

# Instantiate the layout otpimization object
layout_opt = LayoutOptimization(
    fi=fi,
    boundaries=boundaries,
    wd=wd,
    ws=ws,
    freq=freq,
    AEP_initial=AEP_initial,
    opt_options=opt_options,
    min_dist=400
)

# Perform layout optimization
layout_results = layout_opt.optimize()

print("=====================================================")
print("Layout coordinates: ")
for i in range(len(layout_results[0])):
    print(
        "Turbine",
        i,
        ": \tx = ",
        "{:.1f}".format(layout_results[0][i]),
        "\ty = ",
        "{:.1f}".format(layout_results[1][i]),
    )

# Calculate new AEP results
fi.reinitialize_flow_field(layout_array=(layout_results[0], layout_results[1]))
AEP_optimized = fi.get_farm_AEP(wd, ws, freq)

print("=====================================================")
print("Total AEP Gain = %.1f%%" % (100.0 * (AEP_optimized - AEP_initial) / AEP_initial))
print("=====================================================")

# Plot the new layout vs the old layout
layout_opt.plot_layout_opt_results()
plt.show()