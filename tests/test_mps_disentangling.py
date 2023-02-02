from src.encoding.mps_encoding import (
    is_unitary,
    get_unitary_form_of_mps_site,
    encode_bond_dimension_two_mps_as_quantum_circuit,
)

from src.disentangling.mps_disentangling import (
    get_matrix_product_disentangler,
    disentangle_mps,
)
import pytest
import numpy as np
import copy
import cirq
from src.mps.mps import get_random_mps, get_wavefunction, get_truncated_mps

SEED = 1234
np.random.seed(SEED)


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
        assert is_unitary(mpd_site)
        assert np.array_equal(get_unitary_form_of_mps_site(mps_site).T.conj(), mpd_site)


# @pytest.mark.parametrize("number_of_sites", range(2, 20, 3))
# def test_disentangle_mps_does_not_destroy_input(number_of_sites):
#     mps = get_random_mps(number_of_sites, 2)
#     copied_mps = copy.deepcopy(mps)
#     _ = disentangle_mps(mps)
#     assert np.array_equal(copied_mps, mps)


# @pytest.mark.parametrize("number_of_sites", range(2, 20, 3))
# @pytest.mark.parametrize("max_bond_dimension", range(2, 1000, 57))
# def test_disentangle_mps_returns_mps(number_of_sites, max_bond_dimension):
#     mps = get_random_mps(number_of_sites, max_bond_dimension)
#     disentangled_mps = disentangle_mps(mps)
#     assert len(disentangled_mps) == len(mps)
#     assert disentangled_mps[0].shape == (1, 2, 2)
#     assert disentangled_mps[-1].shape == (2, 2, 1)
#     for i, site in enumerate(disentangled_mps):
#         assert len(site.shape) == 3
#         assert site.shape[1] == 2
#         assert site.shape[0] <= min(
#             int(2 ** i), int(2 ** (number_of_sites - i)), max_bond_dimension
#         )
#         assert site.shape[2] <= min(
#             int(2 ** (i + 1)), int(2 ** (number_of_sites - i - 1)), max_bond_dimension
#         )


# @pytest.mark.parametrize("unentangled_state", [])
# def test_disentangle_mps_leaves_unentangled_state_alone(unentangled_state):
#     unentangled_mps = get_mps(state)
#     disentangled_mps = disentangle_mps(unentangled_mps)
#     assert np.array_equal(unentangled_mps, disentangled_mps)


# @pytest.mark.parametrize(
#     "bell_state",
#     [
#         (1 / np.sqrt(2)) * [1.0, 0.0, 0.0, 1.0],
#         (1 / np.sqrt(2)) * [1.0, 0.0, 1.0, 0.0],
#         (1 / np.sqrt(2)) * [1.0, 1.0, 0.0, 0.0],
#         (1 / np.sqrt(2)) * [0.0, 1.0, 0.0, 1.0],
#         (1 / np.sqrt(2)) * [0.0, 1.0, 1.0, 0.0],
#         (1 / np.sqrt(2)) * [0.0, 0.0, 1.0, 1.0],
#         (1 / np.sqrt(2)) * [1.0, 0.0, 0.0, -1.0],
#         (1 / np.sqrt(2)) * [1.0, 0.0, -1.0, 0.0],
#         (1 / np.sqrt(2)) * [1.0, -1.0, 0.0, 0.0],
#         (1 / np.sqrt(2)) * [0.0, 1.0, 0.0, -1.0],
#         (1 / np.sqrt(2)) * [0.0, 1.0, -1.0, 0.0],
#         (1 / np.sqrt(2)) * [0.0, 0.0, 1.0, -1.0],
#     ],
# )
# def test_disentangle_mps_completely_disentangles_two_qubit_bell_state(bell_state):
#     bell_state_mps = get_mps(bell_state)
#     disentangled_bell_state_mps = disentangle_mps(bell_state_mps)
#     entenglement = np.abs(
#         2
#         * (
#             np.abs(disentangled_bell_state_mps[0]) ** 2
#             * np.abs(disentangled_bell_state_mps[3]) ** 2
#             - np.abs(disentangled_bell_state_mps[1]) ** 2
#             * np.abs(disentangled_bell_state_mps[2]) ** 2
#         )
#     )
#     assert np.isclose(entenglement, 0, 1e-15)


# @pytest.mark.parametrize("number_of_sites", range(2, 20, 3))
# @pytest.mark.parametrize("max_bond_dimension", range(2, 1000, 57))
# def test_disentangle_mps_always_returns_mps_with_smaller_bond_dimension(
#     number_of_sites, max_bond_dimension
# ):
#     mps = get_random_mps(number_of_sites, max_bond_dimension)
#     disentangled_mps = disentangle_mps(mps)
#     disentangled_wf = get_wavefunction(disentangled_mps)

#     truncated_disentangled_mps = get_truncated_mps(
#         disentangled_mps, int(max_bond_dimension / 2)
#     )
#     truncated_disentangled_wf = get_wavefunction(truncated_disentangled_mps)

#     assert np.testing.assert_array_almost_equal(
#         disentangled_wf, truncated_disentangled_wf, 16
#     )
#     assert np.testing.assert_array_almost_equal(
#         disentangled_mps, truncated_disentangled_mps, 16
#     )
#     cirq.testing.assert_allclose_up_to_global_phase(
#         disentangled_wf, truncated_disentangled_wf, rtol=1e-2, atol=1e-2
#     )
