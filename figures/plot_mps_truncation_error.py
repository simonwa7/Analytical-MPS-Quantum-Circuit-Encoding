from mps.mps import get_random_mps, get_truncated_mps, get_wavefunction
import numpy as np
import matplotlib.pyplot as plt
import copy

truncated_bond_dimensions = range(1, 1024, 1)
approximation_errors = []

number_of_sites = 20
maximum_bond_dimension = 512
random_mps = get_random_mps(number_of_sites, maximum_bond_dimension)
random_mps_wf = get_wavefunction(random_mps)

for truncated_bond_dimension in truncated_bond_dimensions:
    truncated_mps = get_truncated_mps(
        copy.deepcopy(random_mps), truncated_bond_dimension
    )
    truncated_mps_wf = get_wavefunction(truncated_mps)
    approximation_error = np.linalg.norm(random_mps_wf - truncated_mps_wf) / np.sqrt(2)
    print(approximation_error)
    approximation_errors.append(approximation_error)
plt.rc("font", size=10)  # controls default text size
plt.rc("axes", titlesize=10)  # fontsize of the title
plt.rc("axes", labelsize=10)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=10)  # fontsize of the x tick labels
plt.rc("ytick", labelsize=10)  # fontsize of the y tick labels
plt.rc("legend", fontsize=10)  # fontsize of the legend
ax = plt.figure(
    figsize=(8, 8),
)
plt.xlabel("Truncated Bond Dimension")
plt.ylabel("Error in Approximation")
plt.title(
    "Truncation Error for Random MPS\nwith {} Sites and Maximum Bond Dimension {}".format(
        number_of_sites, maximum_bond_dimension
    )
)
plt.plot(
    truncated_bond_dimensions,
    approximation_errors,
)
plt.yscale("log")
plt.xscale("log", base=2)
plt.show()
