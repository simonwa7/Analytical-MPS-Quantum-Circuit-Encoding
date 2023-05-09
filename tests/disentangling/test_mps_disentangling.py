from qcmps.encoding._kak_decomposition import _is_unitary, _get_unitary_form_of_mps_site
from qcmps.encoding.mps_encoding import (
    encode_bond_dimension_two_mps_as_quantum_circuit,
    encode_mps_in_quantum_circuit,
)

from qcmps.disentangling.mps_disentangling import (
    get_matrix_product_disentangler,
    disentangle_mps,
)
from qcmps.mps.mps import (
    get_random_mps,
    get_wavefunction,
    get_truncated_mps,
    get_mps,
    _get_disentangler_from_circuit,
)
import pytest
import numpy as np
from ncon import ncon
import copy

SEED = 1234
np.random.seed(SEED)
np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


@pytest.mark.parametrize("number_of_sites", range(2, 20, 3))
def test_get_matrix_product_disentangler_does_not_destroy_input(number_of_sites):
    mps = get_random_mps(number_of_sites, 2)
    copied_mps = copy.deepcopy(mps)
    _ = get_matrix_product_disentangler(mps)
    for copied_site, site in zip(copied_mps, mps):
        assert np.array_equal(copied_site, site)


@pytest.mark.parametrize("number_of_sites", range(2, 20, 3))
def test_get_matrix_product_disentangler_returns_conjugated_mps_sites(number_of_sites):
    mps = get_random_mps(number_of_sites, 2)
    mpd = get_matrix_product_disentangler(mps)
    for mps_site, mpd_site in zip(mps, mpd):
        assert _is_unitary(mpd_site)
        assert np.array_equal(
            _get_unitary_form_of_mps_site(mps_site).T.conj(), mpd_site
        )


@pytest.mark.parametrize("number_of_sites", range(2, 4, 1))
def test_disentangle_mps_does_not_destroy_input(number_of_sites):
    mps = get_random_mps(number_of_sites, 2)
    copied_mps = copy.deepcopy(mps)
    mpd = get_matrix_product_disentangler(get_truncated_mps(mps, 2))
    copied_mpd = copy.deepcopy(mpd)
    _ = disentangle_mps(mps, mpd)
    for copied_site, site in zip(copied_mps, mps):
        assert np.array_equal(copied_site, site)

    for copied_operator, operator in zip(copied_mpd, mpd):
        assert np.array_equal(copied_operator, operator)


@pytest.mark.parametrize("number_of_sites", range(2, 4, 1))
@pytest.mark.parametrize("max_bond_dimension", range(2, 1000, 57))
def test_disentangle_mps_returns_mps(number_of_sites, max_bond_dimension):
    mps = get_random_mps(number_of_sites, max_bond_dimension)
    mpd = get_matrix_product_disentangler(get_truncated_mps(mps, 2))
    disentangled_mps = disentangle_mps(mps, mpd)
    assert len(disentangled_mps) == len(mps)
    assert disentangled_mps[0].shape == (1, 2, 2)
    assert disentangled_mps[-1].shape == (2, 2, 1)
    for i, site in enumerate(disentangled_mps):
        assert len(site.shape) == 3
        assert site.shape[1] == 2
        assert site.shape[0] <= min(
            int(2**i), int(2 ** (number_of_sites - i)), max_bond_dimension
        )
        assert site.shape[2] <= min(
            int(2 ** (i + 1)), int(2 ** (number_of_sites - i - 1)), max_bond_dimension
        )


@pytest.mark.parametrize(
    "bell_state",
    [
        (1 / np.sqrt(2)) * np.asarray([1.0, 0.0, 0.0, 1.0]),
        (1 / np.sqrt(2)) * np.asarray([1.0, 0.0, 1.0, 0.0]),
        (1 / np.sqrt(2)) * np.asarray([1.0, 1.0, 0.0, 0.0]),
        (1 / np.sqrt(2)) * np.asarray([0.0, 1.0, 0.0, 1.0]),
        (1 / np.sqrt(2)) * np.asarray([0.0, 1.0, 1.0, 0.0]),
        (1 / np.sqrt(2)) * np.asarray([0.0, 0.0, 1.0, 1.0]),
        (1 / np.sqrt(2)) * np.asarray([1.0, 0.0, 0.0, -1.0]),
        (1 / np.sqrt(2)) * np.asarray([1.0, 0.0, -1.0, 0.0]),
        (1 / np.sqrt(2)) * np.asarray([1.0, -1.0, 0.0, 0.0]),
        (1 / np.sqrt(2)) * np.asarray([0.0, 1.0, 0.0, -1.0]),
        (1 / np.sqrt(2)) * np.asarray([0.0, 1.0, -1.0, 0.0]),
        (1 / np.sqrt(2)) * np.asarray([0.0, 0.0, 1.0, -1.0]),
    ],
)
def test_disentangle_mps_completely_disentangles_two_qubit_bell_state(bell_state):
    u, s, v = np.linalg.svd(bell_state.reshape(2, 2), full_matrices=False)
    bell_state_mps = [
        u.reshape(1, 2, 2),
        ncon([np.diag(s), v], [[-1, 1], [1, -2]]).reshape(2, 2, 1),
    ]
    mpd = get_matrix_product_disentangler(get_truncated_mps(bell_state_mps, 2))
    disentangled_bell_state_mps = disentangle_mps(bell_state_mps, mpd)
    disentangled_bell_state = get_wavefunction(disentangled_bell_state_mps)

    zero_state = np.zeros(4)
    zero_state[0] = 1
    overlap = abs(np.dot(disentangled_bell_state.T.conj(), zero_state))
    assert overlap > (1 - 1e-14)


@pytest.mark.parametrize("number_of_sites", range(2, 10, 1))
def test_disentangle_mps_completely_disentangles_mps_with_bond_dimension_2(
    number_of_sites,
):
    mps = get_random_mps(number_of_sites, 2)
    mpd = get_matrix_product_disentangler(get_truncated_mps(mps, 2))
    disentangled_mps = disentangle_mps(mps, mpd)
    disentangled_wf = get_wavefunction(disentangled_mps)

    zero_state = np.zeros(2**number_of_sites)
    zero_state[0] = 1
    overlap = abs(np.dot(disentangled_wf.T.conj(), zero_state))
    assert overlap > (1 - 1e-14)


@pytest.mark.parametrize("number_of_sites", range(2, 10, 1))
def test_disentangle_mps_completely_disentangles_mps_with_bond_dimension_2_naive_with_circuit_method(
    number_of_sites,
):
    mps = get_random_mps(number_of_sites, 2)
    circuit, _ = encode_bond_dimension_two_mps_as_quantum_circuit(mps)
    disentangled_mps = disentangle_mps(mps, circuit, strategy="naive_with_circuit")
    disentangled_wf = get_wavefunction(disentangled_mps)

    zero_state = np.zeros(2**number_of_sites)
    zero_state[0] = 1
    overlap = abs(
        np.dot(
            zero_state.T.conj(),
            disentangled_wf.reshape(2 ** len(mps)),
        )
    )
    assert overlap > (1 - 1e-14)


@pytest.mark.xfail  # Reason here is that the algorithm does not necessitate this condition, but it is often true
@pytest.mark.parametrize("number_of_sites", range(2, 10, 1))
@pytest.mark.parametrize("max_bond_dimension", range(4, 1000, 57))
def test_disentangle_mps_always_increases_overlap_with_zero_state(
    number_of_sites, max_bond_dimension
):
    mps = get_random_mps(number_of_sites, max_bond_dimension)
    mps_wf = get_wavefunction(mps)

    circuit, _, _ = encode_mps_in_quantum_circuit(mps)
    disentangled_mps = disentangle_mps(mps, circuit, strategy="naive_with_circuit")
    disentangled_wf = get_wavefunction(disentangled_mps)

    zero_state = np.zeros(2**number_of_sites)
    zero_state[0] = 1

    mps_overlap = abs(np.dot(mps_wf.T.conj(), zero_state))
    disentangled_overlap = abs(np.dot(disentangled_wf.T.conj(), zero_state))
    assert disentangled_overlap > mps_overlap


@pytest.mark.parametrize("number_of_sites", range(2, 10, 1))
def test_that_matrix_product_disentanglers_for_al_zeros_is_identity(number_of_sites):
    all_zeros = np.zeros(2**number_of_sites, dtype="complex128")
    all_zeros[0] = 1

    mps = get_mps(all_zeros)
    circuit, _, _ = encode_mps_in_quantum_circuit(mps)
    unitary = _get_disentangler_from_circuit(circuit)
    identity = np.eye(2**number_of_sites)
    global_phase = 1 / unitary.T.conj()[0][0]
    np.testing.assert_array_almost_equal(global_phase * unitary.T.conj(), identity, 14)


def test_that_matrix_product_disentanglers_for_full_circuit_are_consistent_up_to_a_global_phase():
    number_of_sites = 4
    all_zeros = np.zeros(2**number_of_sites, dtype="complex128")
    all_zeros[0] = 1
    global_phase_all_zeros = np.dot(all_zeros, np.exp(1.1j))

    mps = get_mps(all_zeros)
    circuit, _, _ = encode_mps_in_quantum_circuit(mps)
    unitary = _get_disentangler_from_circuit(circuit)

    gp_mps = get_mps(global_phase_all_zeros)
    gp_circuit, _, _ = encode_mps_in_quantum_circuit(gp_mps)
    gp_unitary = _get_disentangler_from_circuit(gp_circuit)

    np.testing.assert_array_almost_equal(unitary, gp_unitary, 14)
    # identity = np.eye(2 ** number_of_sites)
    # global_phase = 1 / unitary.T.conj()[0][0]
    # np.testing.assert_array_almost_equal(global_phase * unitary.T.conj(), identity, 14)
