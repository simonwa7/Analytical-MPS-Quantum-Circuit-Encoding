from qcmps.mps.mps import (
    get_random_mps,
    truncate_singular_values,
    get_truncated_mps,
    get_wavefunction,
    _contract_and_decompose_and_truncate_sites_into_left_canonical_form,
    get_mps,
)
import pytest
import numpy as np
import copy

SEED = 1234
np.random.seed(SEED)
np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


@pytest.mark.parametrize("number_of_sites", range(2, 20, 2))
@pytest.mark.parametrize("input_max_bond_dimension", range(2, 1000, 17))
def test_shape_get_random_mps_even(number_of_sites, input_max_bond_dimension):
    max_possible_bond_dimension = int(2 ** int((number_of_sites / 2)))

    mps = get_random_mps(number_of_sites, max_bond_dimension=input_max_bond_dimension)

    assert mps[0].shape == (1, 2, 2)
    assert mps[-1].shape == (2, 2, 1)

    for i, site in enumerate(mps[: int(number_of_sites / 2) - 1]):
        assert len(site.shape) == 3
        assert site.shape[0] <= max_possible_bond_dimension
        assert site.shape[1] == 2
        assert site.shape[2] <= max_possible_bond_dimension
        assert mps[-i - 1].shape[0] <= max_possible_bond_dimension
        assert mps[-i - 1].shape[1] == 2
        assert mps[-i - 1].shape[2] <= max_possible_bond_dimension

        bond_index = int(i % (number_of_sites / 2))
        left_bond_dimension = min(input_max_bond_dimension, 2**bond_index)
        right_bond_dimension = min(input_max_bond_dimension, 2 ** (bond_index + 1))
        assert site.shape == (left_bond_dimension, 2, right_bond_dimension)
        assert mps[-i - 1].shape == (right_bond_dimension, 2, left_bond_dimension)


@pytest.mark.parametrize("number_of_sites", range(3, 20, 2))
@pytest.mark.parametrize("input_max_bond_dimension", range(2, 1000, 17))
def test_shape_get_random_mps_odd(number_of_sites, input_max_bond_dimension):
    max_possible_bond_dimension = int(2 ** int((number_of_sites / 2)))

    mps = get_random_mps(number_of_sites, max_bond_dimension=input_max_bond_dimension)

    assert mps[0].shape == (1, 2, 2)
    assert mps[-1].shape == (2, 2, 1)

    for i, site in enumerate(mps[: int(number_of_sites / 2) - 1]):
        assert len(site.shape) == 3
        assert site.shape[0] <= max_possible_bond_dimension
        assert site.shape[1] == 2
        assert site.shape[2] <= max_possible_bond_dimension
        assert mps[-i - 1].shape[0] <= max_possible_bond_dimension
        assert mps[-i - 1].shape[1] == 2
        assert mps[-i - 1].shape[2] <= max_possible_bond_dimension

        bond_index = int(i % (number_of_sites / 2))
        left_bond_dimension = min(input_max_bond_dimension, 2**bond_index)
        right_bond_dimension = min(input_max_bond_dimension, 2 ** (bond_index + 1))
        assert site.shape == (left_bond_dimension, 2, right_bond_dimension)
        assert mps[-i - 1].shape == (right_bond_dimension, 2, left_bond_dimension)

    bond_index = int(number_of_sites / 2)
    site = mps[bond_index]
    assert len(site.shape) == 3
    bond_dimension = min(input_max_bond_dimension, 2**bond_index)
    assert site.shape[0] == bond_dimension
    assert site.shape[1] == 2
    assert site.shape[2] == bond_dimension


# @pytest.mark.parametrize("number_of_sites", range(2, 20, 2))
# # @pytest.mark.parametrize("max_bond_dimension", range(2, 1000, 17))
# def test_get_random_mps_is_in_left_canonical_form(number_of_sites):
#     mps = get_random_mps(number_of_sites, 2)
#     # Assert first site is unitary
#     first_site = mps[0].reshape(mps[0].shape[1], mps[0].shape[2])
#     assert np.allclose(np.eye(2), first_site.dot(first_site.T.conj()))

#     truncated_mps = get_truncated_mps(mps, 8)
#     for index, site in enumerate(truncated_mps):
#         np.testing.assert_array_almost_equal(site, mps[index], 1e-16)
#     assert len(mps) == len(truncated_mps)


@pytest.mark.parametrize("number_of_sites", range(2, 20, 3))
@pytest.mark.parametrize("max_bond_dimension", range(2, 1000, 57))
def test_truncate_mps_is_in_left_canonical_form(number_of_sites, max_bond_dimension):
    mps = get_random_mps(number_of_sites, max_bond_dimension)
    mps = get_truncated_mps(mps, 8)
    # Assert first site is unitary
    first_site = mps[0].reshape(mps[0].shape[1], mps[0].shape[2])
    assert np.allclose(np.eye(2), first_site.dot(first_site.T.conj()))

    mps = get_truncated_mps(mps, 5)
    # Assert first site is unitary
    first_site = mps[0].reshape(mps[0].shape[1], mps[0].shape[2])
    assert np.allclose(np.eye(2), first_site.dot(first_site.T.conj()))

    mps = get_truncated_mps(mps, 2)
    # Assert first site is unitary
    first_site = mps[0].reshape(mps[0].shape[1], mps[0].shape[2])
    assert np.allclose(np.eye(2), first_site.dot(first_site.T.conj()))

    truncated_mps = get_truncated_mps(mps, 8)
    np.testing.assert_almost_equal(
        get_wavefunction(mps), get_wavefunction(truncated_mps), 14
    )
    assert len(mps) == len(truncated_mps)


def test_truncate_singular_values_checks_ordering():
    unsorted_singular_values = np.asarray([0.05, 0.95])
    pytest.raises(AssertionError, truncate_singular_values, unsorted_singular_values, 1)


def test_truncate_singular_values_doesnt_destroy_original_array():
    singular_values = np.asarray([0.95, 0.05])
    copied_singular_values = copy.deepcopy(singular_values)
    truncate_singular_values(singular_values, 1)
    assert np.array_equal(singular_values, copied_singular_values)


@pytest.mark.parametrize("target_number_of_values", range(4, 20, 3))
def test_truncate_singular_values_expected_same_values_back(target_number_of_values):
    singular_values = np.asarray([0.90, 0.05, 0.03, 0.02])
    truncated_singular_values = truncate_singular_values(
        singular_values, target_number_of_values
    )
    assert np.array_equal(singular_values, truncated_singular_values)


@pytest.mark.parametrize("target_number_of_values", range(4, 20, 3))
def test_truncate_singular_values_renormalizes(target_number_of_values):
    singular_values = sorted(np.random.uniform(0, 1, 100))[::-1]
    truncated_singular_values = truncate_singular_values(
        singular_values, target_number_of_values
    )
    assert np.isclose(np.linalg.norm(truncated_singular_values, ord=1), 1, 1e-16)


def test_truncate_singular_values_simple():
    singular_values = np.asarray([0.9, 0.05, 0.03, 0.02])
    truncated_singular_values = truncate_singular_values(singular_values, 2)
    assert np.isclose(truncated_singular_values[0], 0.9 / 0.95, 1e-16)
    assert np.isclose(truncated_singular_values[1], 0.05 / 0.95, 1e-16)


@pytest.mark.parametrize("truncated_bond_dimension", range(2, 256, 17))
def test_get_truncated_mps_even(truncated_bond_dimension):
    mps = get_random_mps(20, max_bond_dimension=1024)
    truncated_mps = get_truncated_mps(mps, truncated_bond_dimension)

    for site in truncated_mps:
        left_bond_dimension = site.shape[0]
        right_bond_dimension = site.shape[2]
        assert left_bond_dimension <= truncated_bond_dimension
        assert right_bond_dimension <= truncated_bond_dimension


def test__contract_and_decompose_and_truncate_sites_into_left_canonical_form():
    mps = get_random_mps(4, 8)
    left_site = mps[1]
    right_site = mps[2]
    shared_bond_dimension = left_site.shape[2]
    assert right_site.shape[0] == shared_bond_dimension
    (
        updated_left_site,
        updated_right_site,
    ) = _contract_and_decompose_and_truncate_sites_into_left_canonical_form(
        left_site, right_site, shared_bond_dimension
    )
    np.testing.assert_array_almost_equal(left_site, updated_left_site, 1e-16)
    np.testing.assert_array_almost_equal(right_site, updated_right_site, 1e-16)


def test_get_truncated_mps_doesnt_alter_original_mps():
    mps = get_random_mps(4, max_bond_dimension=8)
    copied_mps = copy.deepcopy(mps)
    get_truncated_mps(mps, 8)
    for index, site in enumerate(copied_mps):
        assert np.array_equal(site, mps[index])
    assert len(mps) == len(copied_mps)


def test_get_truncated_mps_gives_back_same_state():
    mps = get_random_mps(4, max_bond_dimension=8)
    mps = get_truncated_mps(mps, 8)
    truncated_mps = get_truncated_mps(mps, 8)
    for index, site in enumerate(truncated_mps):
        np.testing.assert_array_almost_equal(site, mps[index], 1e-16)
    assert len(mps) == len(truncated_mps)


def test_repeated_truncations_gives_same_result():
    for i in range(50):
        np.random.seed(i)
        mps = get_random_mps(4, max_bond_dimension=8)
        truncated_mps = get_truncated_mps(mps, 4)
        for _ in range(30):
            new_truncated_mps = get_truncated_mps(truncated_mps, 4)
            for index, site in enumerate(new_truncated_mps):
                np.testing.assert_array_almost_equal(
                    abs(truncated_mps[index]), abs(site), 1e-16
                )


@pytest.mark.parametrize("number_of_sites", range(2, 20, 1))
def test_get_wavefunction_size(number_of_sites):
    mps = get_random_mps(number_of_sites, max_bond_dimension=1024)
    wavefunction = get_wavefunction(mps)

    assert wavefunction.shape == ((2**number_of_sites),)


@pytest.mark.parametrize("number_of_sites", range(2, 20, 1))
def test_get_wavefunction_normalizes(number_of_sites):
    mps = get_random_mps(number_of_sites, max_bond_dimension=1024)
    wavefunction = get_wavefunction(mps)

    assert np.linalg.norm(wavefunction, ord=2)
    assert np.linalg.norm(wavefunction, ord=1)


def test_get_wavefunction_bell_state():
    bell_state_mps = np.asarray(
        [
            np.asarray(
                [[[1.0, 0.0], [0.0, 1.0]]],
            ),
            np.asarray(
                [[[0.5], [0.0]], [[0.0], [0.5]]],
            ),
        ],
    )
    bell_state = np.asarray([1.0 / np.sqrt(2), 0.0, 0.0, 1.0 / np.sqrt(2)], dtype=float)
    wavefunction = get_wavefunction(bell_state_mps)
    np.testing.assert_array_almost_equal(wavefunction, bell_state, 1e-16)


def test_get_wavefunction_doesnt_destroy_original_mps():
    mps = get_random_mps(4, 8)
    copied_mps = copy.deepcopy(mps)
    get_wavefunction(mps)
    for i, site in enumerate(copied_mps):
        assert np.array_equal(mps[i], site)


@pytest.mark.parametrize("number_of_sites", range(2, 20, 1))
def test_integration_for_mps_truncation_accuracy(number_of_sites):
    for max_bond_dimension in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
        mps = get_random_mps(number_of_sites, max_bond_dimension)
        mps_wf = get_wavefunction(mps)

        accuracies = []
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
            truncated_mps = get_truncated_mps(mps, truncated_bond_dimension)
            truncated_mps_wf = get_wavefunction(truncated_mps)

            one_norm = np.linalg.norm(mps_wf - truncated_mps_wf, ord=1)
            two_norm = np.linalg.norm(mps_wf - truncated_mps_wf, ord=2)

            if truncated_bond_dimension >= max_bond_dimension:
                # Check that truncation is equivalent if truncated_bond_dimension
                # is greater than max_bond_dimension
                assert np.isclose(
                    one_norm,
                    0,
                    atol=1e-10,
                )
                assert np.isclose(
                    two_norm,
                    0,
                    atol=1e-12,
                )

            if truncated_bond_dimension >= max_bond_dimension / 1.5:
                # Check that truncation is close if truncated_bond_dimension
                # is greater than max_bond_dimension / 2
                assert np.isclose(one_norm, 0, atol=0.4)
                assert np.isclose(two_norm, 0, atol=0.2)

            for accuracy in accuracies:
                # Check that truncation with higher bond dimension produces
                # at least as good of an approximation
                assert two_norm <= accuracy

            accuracies.append(one_norm)


@pytest.mark.parametrize("tensor_decomposition_method", ["qr", "svd"])
@pytest.mark.parametrize("number_of_sites", range(2, 15, 1))
def test_get_mps_size(number_of_sites, tensor_decomposition_method):
    wavefunction = np.random.uniform(0, 1, 2**number_of_sites)
    wavefunction /= sum(wavefunction)
    mps = get_mps(wavefunction, tensor_decomposition_method=tensor_decomposition_method)

    assert len(mps) == number_of_sites


@pytest.mark.parametrize("tensor_decomposition_method", ["qr", "svd"])
def test_get_mps_bell_state(tensor_decomposition_method):
    expected_mps = np.asarray(
        [
            np.asarray(
                [[[1.0, 0.0], [0.0, 1.0]]],
            ),
            np.asarray(
                [[[0.5], [0.0]], [[0.0], [0.5]]],
            ),
        ],
    )
    bell_state = np.asarray([1.0 / np.sqrt(2), 0.0, 0.0, 1.0 / np.sqrt(2)], dtype=float)
    bell_state_mps = get_mps(
        bell_state, tensor_decomposition_method=tensor_decomposition_method
    )
    assert len(bell_state_mps) == len(expected_mps)
    for site, expected_site in zip(bell_state_mps, expected_mps):
        np.testing.assert_almost_equal(site, expected_site, 16)


@pytest.mark.parametrize("number_of_sites", range(2, 15, 1))
@pytest.mark.parametrize("tensor_decomposition_method", ["qr", "svd"])
def test_get_mps_doesnt_destroy_original_mps(
    number_of_sites, tensor_decomposition_method
):
    wavefunction = np.random.uniform(0, 1, 2**number_of_sites)
    wavefunction /= sum(wavefunction)
    copied_wavefunction = copy.deepcopy(wavefunction)
    get_mps(wavefunction, tensor_decomposition_method=tensor_decomposition_method)
    assert np.array_equal(wavefunction, copied_wavefunction)


@pytest.mark.parametrize("number_of_sites", range(2, 15, 1))
@pytest.mark.parametrize("tensor_decomposition_method", ["qr", "svd"])
def test_get_mps_integration_test(number_of_sites, tensor_decomposition_method):
    random_mps = get_random_mps(number_of_sites, max_bond_dimension=1024)
    wavefunction = get_wavefunction(random_mps)
    recovered_mps = get_mps(
        wavefunction, tensor_decomposition_method=tensor_decomposition_method
    )
    assert len(random_mps) == len(recovered_mps)

    recovered_wavefunction = get_wavefunction(recovered_mps)
    assert wavefunction.shape == recovered_wavefunction.shape
    np.testing.assert_array_almost_equal(wavefunction, recovered_wavefunction, 14)


def test_get_mps_zero_state_qr_decomp():
    expected_mps = np.array(
        [
            np.array([[[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]]]),
            np.array(
                [
                    [
                        [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    ],
                    [
                        [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
                    ],
                ]
            ),
            np.array(
                [
                    [
                        [
                            1.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            1.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                    ],
                    [
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            1.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            1.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                    ],
                    [
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            1.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            1.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                    ],
                    [
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            1.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            1.0 + 0.0j,
                        ],
                    ],
                ]
            ),
            np.array(
                [
                    [
                        [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    ],
                    [
                        [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
                    ],
                    [
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    ],
                    [
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    ],
                    [
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    ],
                    [
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    ],
                    [
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    ],
                    [
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    ],
                ]
            ),
            np.array(
                [
                    [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]],
                    [[0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]],
                    [[0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]],
                    [[0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]],
                ]
            ),
            np.array([[[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]]]).reshape(
                2, 2, 1
            ),
        ]
    )
    zero_state = np.zeros(2**6)
    zero_state[0] = 1
    zero_state_mps = get_mps(zero_state, tensor_decomposition_method="qr")
    assert len(zero_state_mps) == len(expected_mps)
    for site, expected_site in zip(zero_state_mps, expected_mps):
        np.testing.assert_almost_equal(site, expected_site, 16)


def test_integration_test_separable_state_can_be_reduced_to_bond_dimension_1_with_perfect_fidelity():
    original_state = np.zeros(2**6)
    original_state[5] = 1

    mps = get_mps(original_state)
    truncated_mps = get_truncated_mps(mps, 1)

    truncated_state = get_wavefunction(truncated_mps)

    fidelity = abs(np.dot(original_state.T.conj(), truncated_state))
    np.testing.assert_almost_equal(fidelity, 1, 14)
