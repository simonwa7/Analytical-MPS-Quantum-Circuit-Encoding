from mps.mps import get_random_mps, get_truncated_mps, get_wavefunction
import copy
import numpy as np
from encoding.mps_encoding import encode_bond_dimension_two_mps_as_quantum_circuit

MAX_NUMBER_OF_SITES = 20
for number_of_sites in range(2, MAX_NUMBER_OF_SITES)[::-1]:
    print("----TESTING MPS TRUNCATION WITH {} SITES----".format(number_of_sites))
    for max_bond_dimension in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
        if max_bond_dimension > 2 ** np.floor(number_of_sites / 2):
            continue
        print("# Max Bond Dimension: ", max_bond_dimension)
        mps = get_random_mps(number_of_sites, max_bond_dimension)
        # mps = get_truncated_mps(copy.deepcopy(mps), max_bond_dimension)
        mps_wf = get_wavefunction(copy.deepcopy(mps))

        for truncated_bond_dimension in [
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
        ]:
            if truncated_bond_dimension > max_bond_dimension:
                continue

            truncated_mps = get_truncated_mps(
                copy.deepcopy(mps), truncated_bond_dimension
            )
            truncated_mps_wf = get_wavefunction(truncated_mps)

            print(
                "    truncated bond dimension {}: ".format(truncated_bond_dimension),
                np.linalg.norm(mps_wf - truncated_mps_wf) / np.sqrt(2),
            )

            if truncated_bond_dimension >= max_bond_dimension:
                assert np.isclose(
                    np.linalg.norm(mps_wf - truncated_mps_wf) / np.sqrt(2),
                    0,
                    atol=1e-12,
                )

            if truncated_bond_dimension >= max_bond_dimension / 2:
                assert np.isclose(
                    np.linalg.norm(mps_wf - truncated_mps_wf) / np.sqrt(2), 0, atol=0.2
                )
import cirq

PRINT_WF = False
PRINT_PROBS = False
MAX_NUMBER_OF_SITES = 20
for number_of_sites in range(1, MAX_NUMBER_OF_SITES + 1):
    print("#####Testing QC Conversion for {} Sites#####".format(number_of_sites))

    mps = get_random_mps(number_of_sites, 2, complex=True)
    mps_wf = get_wavefunction(copy.deepcopy(mps))

    circuit, qubits = encode_bond_dimension_two_mps_as_quantum_circuit(mps)

    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
    qc_wf = result.final_state_vector

    if PRINT_WF:
        print("MPS WF: ", mps_wf)
        print("QC WF: ", qc_wf)
    print("WF Norm: ", np.linalg.norm(mps_wf - qc_wf))

    mps_probs = np.abs(mps_wf) ** 2
    qc_probs = np.abs(qc_wf) ** 2
    if PRINT_PROBS:
        print("MPS Probabilities: ", mps_probs)
        print("QC Probabilities: ", qc_probs)
    print("PROBS Norm: ", np.linalg.norm(mps_probs - qc_probs))

from src.disentangling.mps_disentangling import completely_disentangle_bd2_mps
import pdb

mps = get_random_mps(1, 2)
disentangled_mps = completely_disentangle_bd2_mps(mps)
disentangled_mps = get_truncated_mps(disentangled_mps, 4)
truncated_disentangled_mps = get_truncated_mps(copy.deepcopy(disentangled_mps), 2)
fully_truncated_disentangled_mps = get_truncated_mps(copy.deepcopy(disentangled_mps), 1)

mps_wf = get_wavefunction(mps)
disentangled_wf = get_wavefunction(disentangled_mps)
print(disentangled_wf)
truncated_disentangled_mps_wf = get_wavefunction(truncated_disentangled_mps)
fully_truncated_disentangled_mps_wf = get_wavefunction(fully_truncated_disentangled_mps)
print(
    np.linalg.norm(truncated_disentangled_mps_wf - fully_truncated_disentangled_mps_wf)
    / np.sqrt(2)
)

truncated_disentangled_probs = np.abs(truncated_disentangled_mps_wf) ** 2
fully_trtruncated_disentangled_probs = np.abs(fully_truncated_disentangled_mps_wf) ** 2
print(
    np.linalg.norm(truncated_disentangled_probs - fully_trtruncated_disentangled_probs)
)

mps = get_random_mps(2, 2)
disentangled_mps = completely_disentangle_bd2_mps(mps)
disentangled_mps = get_truncated_mps(disentangled_mps, 4)
truncated_disentangled_mps = get_truncated_mps(copy.deepcopy(disentangled_mps), 2)
fully_truncated_disentangled_mps = get_truncated_mps(copy.deepcopy(disentangled_mps), 1)

mps_wf = get_wavefunction(mps)
disentangled_wf = get_wavefunction(disentangled_mps)
truncated_disentangled_mps_wf = get_wavefunction(truncated_disentangled_mps)
fully_truncated_disentangled_mps_wf = get_wavefunction(fully_truncated_disentangled_mps)
print(
    np.linalg.norm(truncated_disentangled_mps_wf - fully_truncated_disentangled_mps_wf)
    / np.sqrt(2)
)

truncated_disentangled_probs = np.abs(truncated_disentangled_mps_wf) ** 2
fully_trtruncated_disentangled_probs = np.abs(fully_truncated_disentangled_mps_wf) ** 2
print(
    np.linalg.norm(truncated_disentangled_probs - fully_trtruncated_disentangled_probs)
)
