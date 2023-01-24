from src.encoding.mps_encoding import (
    is_unitary,
    get_unitary_form_of_mps_site,
    encode_bond_dimension_two_mps_as_quantum_circuit,
    get_kak_decomposition_parameters_for_unitary,
)
import pytest
import numpy as np
import copy
from src.mps.mps import get_random_mps, get_wavefunction

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
    mps = get_random_mps(number_of_sites, 8, complex=True)
    for i, site in enumerate(mps):
        unitary = get_unitary_form_of_mps_site(site)
        assert is_unitary(unitary, atol=1e-1)
        if i > 0:
            import cirq

            cirq.linalg.kak_decomposition(unitary)


def test_get_unitary_form_of_mps_site_returns_square_matrix():
    pass


def test_get_unitary_form_of_mps_site_doesnt_destory_input():
    pass


def test_get_kak_decomposition_parameters_for_unitary():
    # Step 1: get unitary
    # Step 2: get params
    # 3: build unitary
    # 4: check that result is within error of input unitary
    pass


def test_get_kak_decomposition_parameters_for_unitary_fails_for_unitaries_not_4by4():
    pass


def test_get_kak_decomposition_parameters_for_unitary_doesnt_destroy_input():
    pass


def test_encode_bond_dimension_two_mps_as_quantum_circuit():
    # Randomly generate a bunch of bond_dimension 2 MPS's
    # Generate + sim circuits
    # Assert that wf is consistent (up to global phase?)
    pass


def test_encode_bond_dimension_two_mps_as_quantum_circuit_doesnt_destroy_input():
    # Randomly generate a bunch of bond_dimension 2 MPS's
    # Generate + sim circuits
    # Assert that wf is consistent (up to global phase?)
    pass


def test_encode_bond_dimension_two_mps_as_quantum_circuit_raises_assertion_when_bond_dimension_is_not_2():
    # Randomly generate a bunch of bond_dimension 2 MPS's
    # Generate + sim circuits
    # Assert that wf is consistent (up to global phase?)
    pass


@pytest.mark.parametrize("number_of_sites", range(1, 15))
def test_encode_bond_dimension_two_mps_as_quantum_circuit_integration_test(
    number_of_sites,
):
    # Randomly generate a bunch of MPS's for varying bond dimension
    # Generate + sim circuits
    # Assert that wf's get closer to MPS state as bd decreases
    import cirq

    mps = get_random_mps(number_of_sites, 2, complex=True)
    mps_wf = get_wavefunction(copy.deepcopy(mps))
    # is_unitary(get_unitary_form_of_mps_site(mps[1]))
    circuit, qubits = encode_bond_dimension_two_mps_as_quantum_circuit(mps)

    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
    qc_wf = result.final_state_vector

    print("WF Norm: ", np.linalg.norm(mps_wf - qc_wf))
    cirq.testing.assert_allclose_up_to_global_phase(mps_wf, qc_wf, atol=1e-5)
    np.testing.assert_allclose(np.abs(mps_wf) ** 2, np.abs(qc_wf) ** 2)
