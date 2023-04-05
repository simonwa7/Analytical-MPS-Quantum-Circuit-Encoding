from ..encoding.mps_encoding import get_unitary_form_of_mps_site
import pdb
import numpy as np
from ncon import ncon
import copy
import cirq


def disentangle_mps(mps, mpd, strategy="naive"):
    if strategy == "naive":
        return _disentangle_mps_naive(mps, mpd)
    if strategy == "naive_with_circuit":
        return _disentangle_mps_naive_with_circuit(mps, mpd)
    elif strategy == "tensor":
        return _disentangle_mps_tensor(mps, mpd)


def _disentangle_mps_naive(mps, mpd):
    from src.mps.mps import get_wavefunction, get_truncated_mps, get_mps
    from src.encoding.mps_encoding import (
        encode_bond_dimension_two_mps_as_quantum_circuit,
    )

    mps = copy.deepcopy(mps)

    wavefunction = get_wavefunction(mps)
    circuit, _ = encode_bond_dimension_two_mps_as_quantum_circuit(
        get_truncated_mps(mps, 2)
    )
    disentangling_operator = cirq.unitary(circuit).T.conj()
    return get_mps((disentangling_operator @ wavefunction).reshape(len(wavefunction)))


def _disentangle_mps_naive_with_circuit(mps, circuit):
    from src.mps.mps import get_wavefunction, get_mps

    mps_wf = get_wavefunction(mps)
    disentangling_operator = cirq.unitary(circuit).T.conj()
    return get_mps((disentangling_operator @ mps_wf).reshape(len(mps_wf)))


def _disentangle_mps_tensor(mps, mpd):
    pass


def get_matrix_product_disentangler(mps):
    mps = copy.deepcopy(mps)
    mpd = [None] * len(mps)
    for i, mps_site in enumerate(mps):
        mpd[i] = get_unitary_form_of_mps_site(mps_site).T.conj()
    return mpd


def _get_operator_from_mpo(mpo):
    """Contract an mpo to get the full unitary matrix representation with shape (2**number_of_sites, 2**number_of_sites)"""
    mpo = copy.deepcopy(mpo)
    operator = mpo[-1].reshape(2, 2, 2, 2)

    for current_operator in mpo[:-1][::-1]:
        if current_operator.shape == (4, 4):
            current_operator = current_operator.reshape(2, 2, 2, 2)
        operator = current_operator @ operator

    return (operator / np.linalg.norm(operator, ord=2)).reshape(
        2 ** len(mpo), 2 ** len(mpo)
    )
