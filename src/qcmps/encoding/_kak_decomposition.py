import numpy as np
import cirq
from scipy.linalg import null_space, qr
import copy


def _is_unitary(unitary, **kwargs):
    return np.allclose(np.eye(len(unitary)), unitary.dot(unitary.T.conj()), **kwargs)


def _get_unitary_form_of_mps_site(mps_site):
    """Only intended to work for sites with bond dimension 2"""
    assert (
        (mps_site.shape == (2, 2, 2))
        or (mps_site.shape == (2, 2, 1))
        or (mps_site.shape == (1, 2, 1))
        or (mps_site.shape == (1, 2, 2))
    )

    left_bond_dimension = mps_site.shape[0]
    right_bond_dimension = mps_site.shape[2]

    # Find null space (kernel) of mps site
    if mps_site.shape == (2, 2, 1):
        Q, _ = qr(mps_site.reshape(left_bond_dimension * 2, right_bond_dimension))
    else:
        Q = mps_site.reshape(left_bond_dimension * 2, right_bond_dimension)
    Q_dagger = Q.T.conj()
    X = null_space(Q_dagger)

    # Pad matrix columns with null space vectors to get unitary form
    return np.hstack([Q, X])


def _get_kak_decomposition_parameters_for_unitary(unitary):
    unitary = copy.deepcopy(unitary)
    if int(np.log2(unitary.shape[0])) == 1:
        return cirq.linalg.deconstruct_single_qubit_matrix_into_angles(unitary)

    assert int(np.log2(unitary.shape[0])) == 2

    kak = cirq.linalg.kak_decomposition(unitary)
    b0, b1 = kak.single_qubit_operations_before
    a0, a1 = kak.single_qubit_operations_after
    x, y, z = kak.interaction_coefficients

    b0p = cirq.linalg.deconstruct_single_qubit_matrix_into_angles(b0)
    b1p = cirq.linalg.deconstruct_single_qubit_matrix_into_angles(b1)
    a0p = cirq.linalg.deconstruct_single_qubit_matrix_into_angles(a0)
    a1p = cirq.linalg.deconstruct_single_qubit_matrix_into_angles(a1)

    return [*b0p] + [*b1p] + [x, y, z] + [*a0p] + [*a1p]
