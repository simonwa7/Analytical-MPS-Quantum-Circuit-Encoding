from qcmps.mps.mps import get_random_mps, get_wavefunction, get_mps, get_truncated_mps
from qcmps.encoding.mps_encoding import encode_mps_in_quantum_circuit
import numpy as np
import cirq


def overlap(wf1, wf2):
    if len(wf1.shape) == 1:
        wf1 = wf1.reshape(wf1.shape[0])
        wf2 = wf2.reshape(wf1.shape[0])

    return abs(np.dot(wf1.T.conj(), wf2))


def encode_and_calculate_encoding_overlap(mps, number_of_layers):
    mps_wf = get_wavefunction(mps)
    circuit, _, circuits = encode_mps_in_quantum_circuit(
        mps, number_of_layers=number_of_layers
    )

    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
    qc_wf = result.final_state_vector

    caclulated_overlap = overlap(mps_wf, qc_wf)
    print(number_of_layers, caclulated_overlap)
    return caclulated_overlap


np.random.seed(0)

number_of_sites = 6

# Bell State MPS
bell_state_wf = np.zeros(2**number_of_sites)
bell_state_wf[0] = 1.0 / np.sqrt(2)
bell_state_wf[-1] = 1.0 / np.sqrt(2)
mps = get_mps(bell_state_wf)
mps_wf = bell_state_wf

# Random MPS
mps = get_random_mps(number_of_sites, 2**number_of_sites, complex=False)
mps_wf = get_wavefunction(mps)

zero_state = np.zeros(2**number_of_sites)
zero_state[0] = 1
# mps = get_mps(zero_state)
# mps_wf = get_wavefunction(mps)
# mps = get_truncated_mps(mps, 4)
print("Overlap of Target MPS with Zero State: ", overlap(zero_state, mps_wf))

max_bond_dimension = max([site.shape[2] for site in mps])
numbers_of_layers = [i for i in range(1, 13)]
overlaps = [
    encode_and_calculate_encoding_overlap(mps, number_of_layers)
    for number_of_layers in numbers_of_layers
]

infidelities = [1 - (overlap**2) for overlap in overlaps]

# numbers_of_layers = range(1, 9)
# overlaps = [
#     0.012721002430720572,
#     0.041393172159240994,
#     0.17509330873899825,
#     0.10815759777917139,
#     0.18567635209683292,
#     0.15827793954536942,
#     0.17635644872356993,
#     0.20395077533868824,
# ]
import matplotlib.pyplot as plt

plt.plot(numbers_of_layers, infidelities)
plt.ylabel("Infidelity")
plt.xlabel("Number of Layers")
plt.yscale("log")
plt.ylim(
    1e-4,
    1e-0,
)
plt.show()


# plt.plot(numbers_of_layers, infidelities)
# plt.ylabel("Infidelity")
# plt.xlabel("Number of Layers")
# plt.yscale("log")
# plt.ylim(
#     1e-4,
#     1e-0,
# )
# plt.show()
