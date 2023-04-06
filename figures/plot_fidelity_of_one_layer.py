from src.mps.mps import get_random_mps, get_wavefunction
from src.encoding.mps_encoding import encode_mps_in_quantum_circuit
import numpy as np
import cirq

numbers_of_sites = range(2, 24)
max_bond_dimension = 2 ** 16
mpss = [
    get_random_mps(number_of_sites, max_bond_dimension)
    for number_of_sites in numbers_of_sites
]
overlaps = []
for mps in mpss:
    print(len(mps))
    circuit, _, _ = encode_mps_in_quantum_circuit(mps, number_of_layers=1)

    mps_wf = get_wavefunction(mps)
    mps_wf_norm = abs(np.dot(mps_wf.T.conj(), mps_wf))
    simulator = cirq.Simulator()
    prepared_state = simulator.simulate(circuit).final_state_vector
    overlaps.append(abs(np.dot(prepared_state.T.conj(), mps_wf)))

import matplotlib.pyplot as plt

plt.scatter(numbers_of_sites, overlaps)
plt.yscale("linear")
plt.ylim(0, 1.1)
plt.xlim(1, 16)
plt.hlines(1, 0, 20, color="red", ls="dashed")
plt.show()
