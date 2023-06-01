from qcmps.encoding.mps_encoding import (
    is_unitary,
    get_unitary_form_of_mps_site,
    encode_bond_dimension_two_mps_as_quantum_circuit,
    encode_mps_in_quantum_circuit,
)
from qcmps.disentangling.mps_disentangling import disentangle_mps
import pytest
import numpy as np
import copy
from itertools import combinations
import math
import cirq
from qcmps.mps.mps import get_random_mps, get_wavefunction, get_truncated_mps, get_mps

SEED = 1234
np.random.seed(SEED)


@pytest.mark.parametrize("number_of_sites", range(2, 10))
def test_analytical_encoding_algorithm_works(number_of_sites):
    mps = get_random_mps(number_of_sites, max_bond_dimension=10000000000)
    mps_wf = get_wavefunction(mps)

    circuits = []
    current_disentangled_mps = mps

    bd2_mpses = []
    overlaps = []

    for number_of_truncations in range(20):
        # truncate to bd2
        bd2_mps = get_truncated_mps(current_disentangled_mps, 2)
        bd2_mpses.append(bd2_mps)

        # create bd2 circuit
        U_mps2, _ = encode_bond_dimension_two_mps_as_quantum_circuit(bd2_mps)
        circuits.append(U_mps2)

        # collate all bd2circuits
        full_mps_prep_circuit = cirq.Circuit()
        for partial_circuit in circuits[::-1]:
            full_mps_prep_circuit += partial_circuit

        # apply disentangler of all circuits to original mps
        disentangled_mps = disentangle_mps(
            mps, full_mps_prep_circuit, strategy="naive_with_circuit"
        )
        current_disentangled_mps = disentangled_mps

        overlap = 0
        # bd2_mps_wf = get_wavefunction(bd2_mps)
        # overlap += abs(
        #     np.dot(
        #         mps_wf.reshape(2 ** len(mps)).T.conj(),
        #         bd2_mps_wf.reshape(2 ** len(mps)),
        #     )
        # )
        for bd2_mps in bd2_mpses:
            bd2_mps_wf = get_wavefunction(bd2_mps)
            overlap += abs(
                np.dot(
                    mps_wf.reshape(2 ** len(mps)).T.conj(),
                    bd2_mps_wf.reshape(2 ** len(mps)),
                )
            )
        overlaps.append(overlap)

    import matplotlib.pyplot as plt

    plt.plot(range(1, 21), overlaps)
    # plt.ylim(0, 1)
    plt.show()
