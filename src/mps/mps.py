import numpy as np
from ncon import ncon
import copy


def get_random_mps(number_of_sites, max_bond_dimension, complex=True):
    """Generate a random matrix product state with the input number of sites
    and maximum bond dimension. Bond dimension grows exponentially (beginning with 2)
    until reaching the maximum bond dimension. Uncontracted bonds have dimension 2."""
    if number_of_sites == 1:
        if complex:
            return get_truncated_mps(
                [np.random.rand(1, 2, 1) + np.random.rand(1, 2, 1) * 1j], 1
            )
        else:
            return get_truncated_mps([np.random.rand(1, 2, 1)], 1)
    mps = [0 for _ in range(number_of_sites)]

    # Initialize first site
    if complex:
        mps[0] = (
            np.random.rand(1, 2, min(2, max_bond_dimension))
            + np.random.rand(1, 2, min(2, max_bond_dimension)) * 1j
        )
    else:
        mps[0] = np.random.rand(1, 2, min(2, max_bond_dimension))

    for i in range(1, number_of_sites - 1):
        # left bond dimension is dimension of last index of site at previous index
        left_bond_dimension = mps[i - 1].shape[-1]

        # right bond dimension is calculated to be the minimum of exponential growth from the left, right end, or the maximum bond dimension
        right_bond_dimension = min(
            left_bond_dimension * 2,
            2 ** (number_of_sites - i - 1),
            max_bond_dimension,
        )

        if complex:
            mps[i] = (
                np.random.rand(left_bond_dimension, 2, right_bond_dimension)
                + np.random.rand(left_bond_dimension, 2, right_bond_dimension) * 1j
            )
        else:
            mps[i] = np.random.rand(left_bond_dimension, 2, right_bond_dimension)

    # Initialize the last site
    if complex:
        mps[number_of_sites - 1] = (
            np.random.rand(mps[number_of_sites - 2].shape[-1], 2, 1)
            + np.random.rand(mps[number_of_sites - 2].shape[-1], 2, 1) * 1j
        )
    else:
        mps[number_of_sites - 1] = np.random.rand(
            mps[number_of_sites - 2].shape[-1], 2, 1
        )

    # Put mps in left canonical form and return
    return get_truncated_mps(
        get_truncated_mps(mps, max_bond_dimension), max_bond_dimension
    )


def truncate_singular_values(
    singular_values, target_number_of_values, remove_truncated_values=True
):
    """Truncate a list of singular values by setting the singular values indexed after
    the target number to zero and renormalizing. Assumes list of singular values is already
    ordered."""
    singular_values = copy.deepcopy(singular_values)
    sorted_singular_values = np.asarray(sorted(copy.deepcopy(singular_values))[::-1])
    assert np.array_equal(sorted_singular_values, singular_values)

    if len(singular_values) < target_number_of_values:
        return singular_values

    singular_values[target_number_of_values:] = np.zeros(
        len(singular_values) - target_number_of_values
    )

    if remove_truncated_values:
        singular_values = singular_values[:target_number_of_values]
    singular_values /= sum(singular_values)
    return singular_values


def _check_decomposition(left_site, right_site, original_contracted_two_site_tensor):
    left_site = copy.deepcopy(left_site)
    right_site = copy.deepcopy(right_site)
    original_contracted_two_site_tensor = copy.deepcopy(
        original_contracted_two_site_tensor
    )
    updated_contracted_two_site_tensor = ncon(
        [copy.deepcopy(left_site), copy.deepcopy(right_site)],
        [[-1, 1], [1, -2]],
    )
    distance = np.linalg.norm(
        updated_contracted_two_site_tensor - original_contracted_two_site_tensor, ord=2
    )

    if distance > 1e-12:
        raise RuntimeWarning(
            "Decomposition of current tensors leading to inconsistent decomposition with original tensors. Distance is {}".format(
                distance
            )
        )


def _contract_sites(left_site, right_site):
    left_site = copy.deepcopy(left_site)
    right_site = copy.deepcopy(right_site)

    return ncon([left_site, right_site], [[-1, -2, 1], [1, -3, -4]]).reshape(
        left_site.shape[0] * 2, right_site.shape[2] * 2
    )


def _contract_and_decompose_and_truncate_sites_into_left_canonical_form(
    left_site, right_site, max_bond_dimension, check_decomposition=False
):
    left_bond_dimension = left_site.shape[0]
    right_bond_dimension = right_site.shape[-1]
    shared_bond_dimension = left_site.shape[-1]
    assert shared_bond_dimension == right_site.shape[0]
    contracted_site = _contract_sites(left_site, right_site)

    # SVD
    u, s, v = np.linalg.svd(contracted_site, full_matrices=False)
    # Truncate u, s, and v to max_bond_dimension
    new_shared_bond_dimension = min(
        shared_bond_dimension,
        max_bond_dimension,
    )
    s = truncate_singular_values(
        s, new_shared_bond_dimension, remove_truncated_values=True
    )

    u = u[:, :new_shared_bond_dimension]
    v = v[:new_shared_bond_dimension, :]
    updated_left_site = u
    updated_right_site = np.diag(s) @ v

    if check_decomposition:
        _check_decomposition(
            updated_left_site,
            updated_right_site,
            contracted_site,
        )

    left_site = updated_left_site.reshape(
        left_bond_dimension, 2, new_shared_bond_dimension
    )
    right_site = updated_right_site.reshape(
        new_shared_bond_dimension, 2, right_bond_dimension
    )

    return left_site, right_site


def get_truncated_mps(mps, max_bond_dimension):
    """Truncate a MPS to the input maximum bond dimension. If two sites share a bond
    dimension greater than the maximum bond dimension, they are contracted, decomposed
    via SVD, and then the singular values are truncated to the maximum bond dimension
    before being put into left-canonical form."""
    number_of_sites = len(mps)
    mps = copy.deepcopy(mps)

    for site1_index in range(0, number_of_sites - 1):
        site2_index = site1_index + 1

        (
            mps[site1_index],
            mps[site2_index],
        ) = _contract_and_decompose_and_truncate_sites_into_left_canonical_form(
            mps[site1_index],
            mps[site2_index],
            max_bond_dimension,
            check_decomposition=False,
        )

    # TODO: is this normalized?
    return mps


def get_wavefunction(mps):
    """Contract an mps to get the full wavefunction representation with shape (1, 2**number_of_sites)"""
    mps = copy.deepcopy(mps)
    wavefunction = mps[-1]
    wavefunction = wavefunction.reshape(
        wavefunction.shape[0], wavefunction.shape[1] * wavefunction.shape[2]
    )

    for current_site in mps[:-1][::-1]:
        wavefunction = (current_site @ wavefunction).reshape(
            current_site.shape[0],
            current_site.shape[1] * wavefunction.shape[1],
        )

    return (wavefunction / np.linalg.norm(wavefunction, ord=2)).reshape(2 ** len(mps))


def _decompose_site_svd(tensor, site_left_bond_dim, site_right_bond_dim):
    u, s, v = np.linalg.svd(tensor, full_matrices=False)
    left_site = u.reshape(site_left_bond_dim, 2, site_right_bond_dim)
    tensor = ncon([np.diag(s), v], [[-1, 1], [1, -2]])
    return left_site, tensor


def _decompose_site_qr(tensor, site_left_bond_dim, site_right_bond_dim):
    q_u, r_u = np.linalg.qr(tensor)
    left_site = q_u.reshape(site_left_bond_dim, 2, site_right_bond_dim)
    return left_site, r_u


def get_mps(wavefunction, tensor_decomposition_method="svd"):
    wavefunction = copy.deepcopy(wavefunction)
    number_of_sites = int(np.log2(len(wavefunction)))
    assert wavefunction.shape == (2**number_of_sites,)
    mps = [None] * number_of_sites

    if tensor_decomposition_method == "svd":
        decompose_tensor = _decompose_site_svd
    elif tensor_decomposition_method == "qr":
        decompose_tensor = _decompose_site_qr
    else:
        raise RuntimeError(
            "Unrecognized Tensor Decomposition Method: ", tensor_decomposition_method
        )

    rest = wavefunction.reshape(1, 2**number_of_sites)
    for site_index in range(number_of_sites - 1):
        site_left_bond_dim = min(2 ** (site_index), 2 ** (number_of_sites - site_index))
        site_right_bond_dim = min(
            2 ** (site_index + 1), 2 ** (number_of_sites - site_index - 1)
        )
        rest_current_left_bond_dim = rest.shape[0] * 2
        rest_current_right_bond_dim = int(rest.shape[1] / 2)
        rest = rest.reshape(rest_current_left_bond_dim, rest_current_right_bond_dim)
        mps[site_index], rest = decompose_tensor(
            rest, site_left_bond_dim, site_right_bond_dim
        )

    mps[number_of_sites - 1] = rest.reshape(2, 2, 1)
    return get_truncated_mps(mps, 2**number_of_sites)
