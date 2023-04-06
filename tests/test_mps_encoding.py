from src.encoding.mps_encoding import (
    is_unitary,
    get_unitary_form_of_mps_site,
    encode_bond_dimension_two_mps_as_quantum_circuit,
    encode_mps_in_quantum_circuit,
)
import pytest
import numpy as np
import copy
from itertools import combinations
import math
import cirq
from src.mps.mps import get_random_mps, get_wavefunction, get_truncated_mps, get_mps

SEED = 1234
np.random.seed(SEED)


@pytest.mark.parametrize(
    "matrix",
    [
        np.eye(1),
        np.eye(2),
        np.eye(3),
        np.eye(4),
        np.asarray([[0, 1], [1, 0]]),
        np.asarray([[0, -1j], [1j, 0]]),
        np.asarray([[1, 0], [0, -1]]),
    ],
)
def test_matrix_is_unitary(matrix):
    assert is_unitary(matrix)


@pytest.mark.parametrize(
    "matrix",
    [
        np.zeros((1, 1)),
        np.zeros((2, 2)),
        np.zeros((3, 3)),
        np.zeros((4, 4)),
        (1 / np.sqrt(2)) * np.asarray([[1, 1], [1, 1]]),
        np.asarray([[0, 1j], [0, -1j]]),
    ],
)
def test_matrix_is_not_unitary(matrix):
    assert not is_unitary(matrix)


@pytest.mark.parametrize(
    "matrix",
    [
        np.eye(1),
        np.eye(2),
        np.eye(3),
        np.eye(4),
        np.asarray([[0, 1], [1, 0]]),
        np.asarray([[0, -1j], [1j, 0]]),
        np.asarray([[1, 0], [0, -1]]),
    ],
)
def test_matrix_is_unitary_doesnt_destroy_input(matrix):
    copied_matrix = copy.deepcopy(matrix)
    is_unitary(matrix)
    assert np.array_equal(copied_matrix, matrix)


@pytest.mark.parametrize("number_of_sites", range(3, 15))
def test_get_unitary_form_of_mps_site(number_of_sites):
    # Step 1: test that result is unitary
    # Step 2: test that matrix properly encodes mps site
    mps = get_random_mps(number_of_sites, 2, complex=True)
    for i, site in enumerate(mps):
        unitary = get_unitary_form_of_mps_site(site)
        assert is_unitary(unitary, atol=1e-12)
        assert len(unitary.shape) == 2
        assert unitary.shape[0] == unitary.shape[1]
        assert math.log(unitary.shape[0], 2).is_integer()


def test_get_unitary_form_of_mps_site_doesnt_destory_input():
    mps = get_random_mps(10, 2, complex=True)
    mps_site = mps[np.random.choice(range(10))]
    copied_mps_site = copy.deepcopy(mps_site)
    _ = get_unitary_form_of_mps_site(mps_site)
    assert np.array_equal(mps_site, copied_mps_site)


def test_get_unitary_form_of_mps_site_raises_error_on_not_bd2_site():
    mps = get_random_mps(4, 8, complex=True)
    mps_site = mps[1]
    pytest.raises(AssertionError, get_unitary_form_of_mps_site, mps_site)


@pytest.mark.parametrize("number_of_sites", range(1, 12))
def test_encode_bond_dimension_two_mps_as_quantum_circuit(number_of_sites):
    # Randomly generate a bunch of bond_dimension 2 MPS's
    # Generate + sim circuits
    # Assert that wf is consistent (up to global phase)
    mps = get_random_mps(number_of_sites, 2, complex=True)
    mps_wf = get_wavefunction(copy.deepcopy(mps))
    circuit, _ = encode_bond_dimension_two_mps_as_quantum_circuit(mps)

    zero_state = np.zeros(2 ** number_of_sites)
    zero_state[0] = 1
    prepared_state = cirq.unitary(circuit) @ zero_state
    overlap = abs(np.dot(mps_wf.reshape(2 ** len(mps)).T.conj(), prepared_state))
    assert overlap > (1 - 1e-15)


@pytest.mark.parametrize("number_of_sites", range(12, 22))
def test_encode_bond_dimension_two_mps_as_quantum_circuit_larger_systems(
    number_of_sites,
):
    # Randomly generate a bunch of bond_dimension 2 MPS's
    # Generate + sim circuits
    # Assert that wf is consistent (up to global phase)
    mps = get_random_mps(number_of_sites, 2, complex=True)
    mps_wf = get_wavefunction(copy.deepcopy(mps))
    circuit, _ = encode_bond_dimension_two_mps_as_quantum_circuit(mps)

    simulator = cirq.Simulator()
    prepared_state = np.asarray(simulator.simulate(circuit).final_state_vector)
    overlap = abs(np.dot(mps_wf.reshape(2 ** len(mps)).T.conj(), prepared_state))
    assert overlap > (1 - 1e-6)


@pytest.mark.parametrize("sites", combinations([0, 1, 2, 3], 2))
def test_encode_bond_dimension_two_mps_as_quantum_circuit_bell_state(sites):
    bell_state_wf = np.zeros(4)
    bell_state_wf[sites[0]] = 1.0 / np.sqrt(2)
    bell_state_wf[sites[1]] = 1.0 / np.sqrt(2)
    u, s, v = np.linalg.svd(bell_state_wf.reshape(2, 2), full_matrices=False)
    bell_state_mps = [
        u.reshape(1, 2, 2),
        (np.diag(s) @ v).reshape(2, 2, 1),
    ]

    circuit, _ = encode_bond_dimension_two_mps_as_quantum_circuit(bell_state_mps)

    zero_state = np.zeros(4)
    zero_state[0] = 1
    prepared_state = cirq.unitary(circuit) @ zero_state
    overlap = abs(np.dot(bell_state_wf.T.conj(), prepared_state))

    assert overlap > (1 - 1e-15)


def test_encode_bond_dimension_two_mps_as_quantum_circuit_doesnt_destroy_input():
    mps = get_random_mps(10, 2, complex=True)
    copied_mps = copy.deepcopy(mps)
    _, _ = encode_bond_dimension_two_mps_as_quantum_circuit(mps)
    for site, copied_site in zip(mps, copied_mps):
        assert np.array_equal(site, copied_site)


def test_encode_bond_dimension_two_mps_as_quantum_circuit_raises_assertion_when_bond_dimension_is_not_2():
    mps = get_random_mps(10, 4, complex=True)
    pytest.raises(AssertionError, encode_bond_dimension_two_mps_as_quantum_circuit, mps)


# # @pytest.mark.parametrize("number_of_sites", range(2, 10))
# # def test_encode_mps_in_quantum_circuit_random_mps_gets_better_with_additional_layers(
# #     number_of_sites,
# # ):
# #     # Randomly generate a bunch of arbitrary bd MPS's
# #     # Generate + sim circuits
# #     # Assert that wf is consistent (up to global phase)
# #     mps = get_random_mps(number_of_sites, 2 ** int(number_of_sites / 2), complex=True)
# #     mps_wf = get_wavefunction(copy.deepcopy(mps))
# #     max_bond_dimension = max([site.shape[2] for site in mps])

# #     previous_error = np.inf
# #     for number_of_layers in range(int(max_bond_dimension / 2)):
# #         circuit, _, _ = encode_mps_in_quantum_circuit(
# #             mps, number_of_layers=number_of_layers
# #         )

# #         simulator = cirq.Simulator()
# #         result = simulator.simulate(circuit)
# #         qc_wf = result.final_state_vector

# #         current_error = np.linalg.norm(abs(mps_wf ** 2) - abs(qc_wf ** 2))
# #         assert current_error <= previous_error
# #         previous_error = current_error

# #     # for mps_prob, circuit_prob in zip(abs(mps_wf ** 2), abs(qc_wf ** 2)):
# #     #     assert np.isclose(mps_prob, circuit_prob, atol=1e-2, rtol=1e-2)
# #     # np.testing.assert_allclose(
# #     #     abs(mps_wf ** 2),
# #     #     abs(qc_wf ** 2),
# #     #     atol=1e-6,
# #     # )
# #     # cirq.testing.assert_allclose_up_to_global_phase(mps_wf, qc_wf, rtol=1e-1, atol=1e-1)


# @pytest.mark.parametrize("number_of_sites", [2, 3])
# def test_encode_mps_in_quantum_circuit_maximal_bell_state_for_two_and_three_sites_gives_near_perfect_overlap(number_of_sites):
#     bell_state_wf = np.zeros(2 ** number_of_sites)
#     bell_state_wf[0] = 1.0 / np.sqrt(2)
#     bell_state_wf[-1] = 1.0 / np.sqrt(2)
#     bell_state_mps = get_mps(bell_state_wf)

#     circuit, _ = encode_mps_in_quantum_circuit(bell_state_mps, number_of_layers=1)

#     simulator = cirq.Simulator()
#     result = simulator.simulate(circuit)
#     qc_wf = result.final_state_vector


# @pytest.mark.parametrize("number_of_sites", range(2, 12))
# def test_encode_mps_in_quantum_circuit_maximal_bell_state_gives_near_perfect_overlap(number_of_sites):
#     bell_state_wf = np.zeros(2 ** number_of_sites)
#     bell_state_wf[0] = 1.0 / np.sqrt(2)
#     bell_state_wf[-1] = 1.0 / np.sqrt(2)
#     bell_state_mps = get_mps(bell_state_wf)

#     circuit, _ = encode_mps_in_quantum_circuit(bell_state_mps)

#     simulator = cirq.Simulator()
#     result = simulator.simulate(circuit)
#     qc_wf = result.final_state_vector

#     np.testing.assert_allclose(
#         abs(bell_state_wf ** 2),
#         abs(qc_wf ** 2),
#         atol=1e-5,
#     )
#     cirq.testing.assert_allclose_up_to_global_phase(
#         bell_state_wf, qc_wf, rtol=1e-2, atol=1e-2
#     )


# def test_encode_mps_in_quantum_circuit_doesnt_destroy_input():
#     mps = get_random_mps(10, 1024, complex=True)
#     copied_mps = copy.deepcopy(mps)
#     _, _, _ = encode_mps_in_quantum_circuit(mps)
#     for site, copied_site in zip(mps, copied_mps):
#         assert np.array_equal(site, copied_site)
