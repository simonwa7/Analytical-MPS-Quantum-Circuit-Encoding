import numpy as np
import cirq
from scipy.linalg import null_space, qr
import copy


def encode_mps_in_quantum_circuit(mps, number_of_layers=1):
    from src.mps.mps import get_truncated_mps
    from src.disentangling.mps_disentangling import disentangle_mps

    mps = copy.deepcopy(mps)
    number_of_qubits = len(mps)
    qubits = [cirq.LineQubit(i) for i in range(number_of_qubits)]

    max_bond_dimension = max([site.shape[2] for site in mps])
    circuits = []
    disentangled_mps = mps
    while len(circuits) < number_of_layers:
        if len(circuits) > 0:
            circuit = cirq.Circuit()
            for partial_circuit in circuits[::-1]:
                circuit += partial_circuit
            disentangled_mps = disentangle_mps(
                mps, circuit, strategy="naive_with_circuit"
            )

        # max_bond_dimension = int(max_bond_dimension / 2)
        disentangled_mps = get_truncated_mps(disentangled_mps, max_bond_dimension)

        bd2_mps = get_truncated_mps(disentangled_mps, 2)
        partial_circuit, _ = encode_bond_dimension_two_mps_as_quantum_circuit(bd2_mps)
        circuits.append(partial_circuit)

    circuit = cirq.Circuit()
    for partial_circuit in circuits[::-1]:
        circuit += partial_circuit
    return circuit, qubits, circuits


def is_unitary(unitary, **kwargs):
    return np.allclose(np.eye(len(unitary)), unitary.dot(unitary.T.conj()), **kwargs)


def get_unitary_form_of_mps_site(mps_site):
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


def encode_bond_dimension_two_mps_as_quantum_circuit(mps):
    number_of_qubits = len(mps)
    qubits = [cirq.LineQubit(i) for i in range(number_of_qubits)]
    circuit = cirq.Circuit()

    for qubit0, qubit1, mps_site in zip(
        qubits[:-1][::-1], qubits[1:][::-1], mps[1:][::-1]
    ):
        unitary = get_unitary_form_of_mps_site(mps_site)
        parameters = _get_kak_decomposition_parameters_for_unitary(unitary)

        circuit.append(cirq.rz(parameters[0]).on(qubit0))
        circuit.append(cirq.ry(parameters[1]).on(qubit0))
        circuit.append(cirq.rz(parameters[2]).on(qubit0))

        circuit.append(cirq.rz(parameters[3]).on(qubit1))
        circuit.append(cirq.ry(parameters[4]).on(qubit1))
        circuit.append(cirq.rz(parameters[5]).on(qubit1))

        circuit.append(
            cirq.XXPowGate(exponent=-2 * parameters[6] / np.pi).on(qubit0, qubit1)
        )
        circuit.append(
            cirq.YYPowGate(exponent=-2 * parameters[7] / np.pi).on(qubit0, qubit1)
        )
        circuit.append(
            cirq.ZZPowGate(exponent=-2 * parameters[8] / np.pi).on(qubit0, qubit1)
        )

        circuit.append(cirq.rz(parameters[9]).on(qubit0))
        circuit.append(cirq.ry(parameters[10]).on(qubit0))
        circuit.append(cirq.rz(parameters[11]).on(qubit0))

        circuit.append(cirq.rz(parameters[12]).on(qubit1))
        circuit.append(cirq.ry(parameters[13]).on(qubit1))
        circuit.append(cirq.rz(parameters[14]).on(qubit1))

    # Inserting kak decomp on first qubit to rotate first qubit respective to first tensor site
    unitary = get_unitary_form_of_mps_site(mps[0])
    parameters = _get_kak_decomposition_parameters_for_unitary(unitary)
    circuit.append(cirq.rz(parameters[0]).on(qubits[0]))
    circuit.append(cirq.ry(parameters[1]).on(qubits[0]))
    circuit.append(cirq.rz(parameters[2]).on(qubits[0]))

    return circuit, qubits


def _get_kak_decomposition_parameters_for_unitary(unitary):
    import copy

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
