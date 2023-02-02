from ..encoding.mps_encoding import get_unitary_form_of_mps_site
import pdb
import numpy as np
from ncon import ncon
import copy


def disentangle_mps(mps, mpd):
    pass


def _get_matrix_product_disentangler(mps):
    mps = copy.deepcopy(mps)
    mpd = [None] * len(mps)
    for i, mps_site in enumerate(mps):
        mpd[i] = get_unitary_form_of_mps_site(mps_site).T.conj()
    return mpd


def completely_disentangle_bd2_mps(mps):
    mps = copy.deepcopy(mps)
    mpd = _get_matrix_product_disentangler(mps)

    import pdb

    if len(mps) == len(mpd) == 1:
        # _ are dummy, i is outward
        _, i, _ = mps[0].shape
        _, m, l = mpd[0].shape
        return [ncon([mps[0], mpd[0]], [[-1, 1, -3], [-4, 1, -2]])]

    if len(mps) == len(mpd) == 2:
        # Contract first site in MPS with second site in MPD
        _, i, j = mps[0].shape  # 1, 2, 2
        j, k, _ = mps[1].shape  # 2, 2, 1
        m, l, _ = mpd[0].shape  # 1, 2, 2
        n, m, o, p = mpd[1].shape  # 2, 2, 2, 2
        # m, n, p, o = mpd[1].shape
        # contract i with n
        # resulting order should be _, j, p, o, m
        # mpd[1] = ncon([mps[0], mpd[1]], [[-1, 1, -2], [1, -5, -4, -3]]).reshape(
        #     j, p, o, m
        # )
        # mpd[1] = ncon([mps[0], mpd[1]], [[-1, 1, -2], [-5, 1, -3, -4]]).reshape(
        #     j, p, o, m
        # )
        from itertools import permutations

        # for ordering in permutations([1, 2, -1, -2]):
        #     contracted_mps = ncon([mps[0], mps[1]], [[-1, -2, 1], [1, -3, -4]]).reshape(
        #         i, k
        #     )
        #     new_mps_0 = mpd[0].reshape(1, 2, 2)
        #     new_mps_1 = ncon([contracted_mps, mpd[1]], [[1, 2], ordering]).reshape(
        #         2, 2, 1
        #     )
        #     new_mps = [new_mps_0, new_mps_1]
        #     from mps import get_wavefunction

        #     new_wf = get_wavefunction(new_mps)
        #     utemp, singular_values, vhtemp = np.linalg.svd(
        #         new_wf.reshape(2, 2), full_matrices=False
        #     )
        #     print(ordering, singular_values)

        # min_sv1 = np.inf
        # max_sv0 = 0
        # min_orderings = []
        # contracted_mps = ncon([mps[0], mps[1]], [[-1, -2, 1], [1, -3, -4]]).reshape(
        #     i, k
        # )
        # for mpd_ordering in permutations([1, -3, -4, -5]):
        #     contracted_mpo = ncon(
        #         [mpd[0], mpd[1]], [[-1, -2, 1], mpd_ordering]
        #     ).reshape(2, 2, 2, 2)
        #     for mps_ordering in permutations([1, 2]):
        #         for mpo_ordering in permutations([1, 2, -1, -2]):
        #             wf = ncon(
        #                 [contracted_mps, contracted_mpo], [mps_ordering, mpo_ordering]
        #             ).reshape(4)
        #             utemp, singular_values, vhtemp = np.linalg.svd(
        #                 wf.reshape(2, 2), full_matrices=False
        #             )
        #             if singular_values[1] <= min_sv1:
        #                 print(
        #                     "Entanglement is: ",
        #                     np.abs(
        #                         2
        #                         * (
        #                             np.abs(wf[0]) ** 2 * np.abs(wf[3]) ** 2
        #                             - np.abs(wf[1]) ** 2 * np.abs(wf[2]) ** 2
        #                         )
        #                     ),
        #                     "Smallest Singular Value is ",
        #                     singular_values[1],
        #                 )
        #                 print()
        #                 # assert singular_values[0] >= max_sv0
        #                 max_sv0 = singular_values[0]
        #                 min_sv1 = singular_values[1]
        #                 min_orderings.append({"mps": mps_ordering, "mpo": mpo_ordering})

        # print(min_orderings, max_sv0, min_sv1)
        # for ordering in permutations([-1, -2, -3, -4, 1, 1, 2, 2, 3, 3, 4, 4, 9]):
        #     try:
        #         new_wf = ncon(
        #             [mps[0], mps[1], mpd[0], mpd[1]],
        #             [ordering[0:3], ordering[3:6], ordering[6:9], ordering[9:]],
        #         ).reshape(i * k)

        #         utemp, singular_values, vhtemp = np.linalg.svd(
        #             new_wf.reshape(2, 2), full_matrices=False
        #         )
        #         print(ordering, singular_values)
        #     except RuntimeError as e:
        #         print("Failed: ", ordering, e)

        # # resulting order becomes, jo, m, p
        # mpd[1] = mpd[1].transpose(0, 2, 3, 1).reshape(j * o, m, p)
        # mps[1] = mps[1].reshape(j * k)
        # mpd[1] = ncon([mps[1], mpd[1]], [[1], [1, -1, -2]]).reshape(1, 2, 2)
        # mpd[1] = mpd[1].T.conj()
        # mpd[0] = mpd[0].T.conj()
        return mpd
    # # Contract first site in MPS with second site in MPD
    # contracted_site = ncon([mps[0], mpd[1]], [[-1, 1, -2], [-3, -4, -5, 1]]).reshape(
    #     mpd[1].shape
    # )
    # mpd[1] = contracted_site

    # # Loop over contractions on middle sites from site 1 to site n-2:
    # for i in range(1, len(mps) - 1):
    #     # contract MPS site i with MPD site i+1
    #     mpd[i + 1] = ncon([mps[i], mpd[i + 1]], [[-1, 1, -2], [-3, -4, -5, 1]])

    #     # reshape MPD site i to have right bond dim multiplied by MPS site i left bond dim
    #     h, l, m, n = mpd[i].shape
    #     mpd[i] = mpd[i].transpose(1, 2, 3, 0).reshape(l, m, h * n)
    #     # mpd[i] = mpd[i].transpose(3, 2, 1, 0).reshape(u, r, l * d)

    #     # reshape MPD site i+1 to have left bond dim multiplied by MPS site i left bond dim
    #     h, k, n, p, q = mpd[i + 1].shape
    #     mpd[i + 1] = mpd[i + 1].transpose(1, 2, 0, 3, 4).reshape(k, h * n, p, q)
    #     # mpd[i + 1] = mpd[i + 1].transpose(1, 2, 3, 0, 4).reshape(2, 2, 2, 4)

    # # Reshape MPS site n-1 to have one rank (uncontracted * left bond dim)
    # k, q, r = mps[-1].shape
    # mps[-1] = mps[-1].reshape(k * q * r)
    # # Reshape MPD site n-1 to have rank three (left bond dim, <stuff>, uncontracted dim)
    # k, h, p, q = mpd[-1].shape
    # mpd[-1] = mpd[-1].transpose(1, 2, 0, 3).reshape(h, p, k * q)
    # # Contract MPS site n-1 with MPD site n-1
    # mpd[-1] = ncon([mps[-1], mpd[-1]], [[1], [-1, -2, 1]])
    # h, p = mpd[-1].shape
    # mpd[-1] = mpd[-1].reshape(h, p, 1)

    # return mpd
    # for i in range(len(mps)):
    #     mps_site, mpd_site = mps[i], mpd[i]
    #     mps_left_bond_dim, mps_contraction_bond_dim, mps_right_bond_dim = mps[i].shape
    #     (
    #         mpd_contraction_bond_dim,
    #         mpd_right_bond_dim,
    #         mpd_uncontracted_bond_dim,
    #         mpd_left_bond_dim,
    #     ) = mpd[i].shape

    #     assert mps_contraction_bond_dim == mpd_contraction_bond_dim
    #     # assert mps_left_bond_dim == mpd_left_bond_dim
    #     # assert mps_right_bond_dim == mpd_right_bond_dim

    #     disentangled_site = (
    #         ncon([mps_site, mpd_site], [[-1, 1, -2], [1, -3, -4, -5]])
    #         .transpose(0, 4, 3, 1, 2)
    #         .reshape(
    #             mps_left_bond_dim * mpd_left_bond_dim,
    #             mpd_uncontracted_bond_dim,
    #             mps_right_bond_dim * mpd_right_bond_dim,
    #         )
    #     )

    #     disentangled_mps[i] = disentangled_site
    # import copy

    # lc_disentangled_mps = get_truncated_mps(copy.deepcopy(disentangled_mps), 4)
    # truncated_disentangled_mps = get_truncated_mps(lc_disentangled_mps, 2)
    # pdb.set_trace()
