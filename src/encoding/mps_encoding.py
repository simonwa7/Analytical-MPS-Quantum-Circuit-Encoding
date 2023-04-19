import numpy as np
import cirq
import copy
from ._kak_decomposition import (
    _get_unitary_form_of_mps_site,
    _get_kak_decomposition_parameters_for_unitary,
)


def add_MPS_layer(circuit, qubits, parameters):
    assert len(parameters) == (15 * (len(qubits) - 1)) + 3
    parameter_counter = 0
    for qubit0, qubit1 in zip(qubits[:-1][::-1], qubits[1:][::-1]):

        circuit.append(cirq.rz(parameters[parameter_counter]).on(qubit0))
        circuit.append(cirq.ry(parameters[parameter_counter + 1]).on(qubit0))
        circuit.append(cirq.rz(parameters[parameter_counter + 2]).on(qubit0))

        circuit.append(cirq.rz(parameters[parameter_counter + 3]).on(qubit1))
        circuit.append(cirq.ry(parameters[parameter_counter + 4]).on(qubit1))
        circuit.append(cirq.rz(parameters[parameter_counter + 5]).on(qubit1))

        circuit.append(
            cirq.XXPowGate(exponent=-2 * parameters[parameter_counter + 6] / np.pi).on(
                qubit0, qubit1
            )
        )
        circuit.append(
            cirq.YYPowGate(exponent=-2 * parameters[parameter_counter + 7] / np.pi).on(
                qubit0, qubit1
            )
        )
        circuit.append(
            cirq.ZZPowGate(exponent=-2 * parameters[parameter_counter + 8] / np.pi).on(
                qubit0, qubit1
            )
        )

        circuit.append(cirq.rz(parameters[parameter_counter + 9]).on(qubit0))
        circuit.append(cirq.ry(parameters[parameter_counter + 10]).on(qubit0))
        circuit.append(cirq.rz(parameters[parameter_counter + 11]).on(qubit0))

        circuit.append(cirq.rz(parameters[parameter_counter + 12]).on(qubit1))
        circuit.append(cirq.ry(parameters[parameter_counter + 13]).on(qubit1))
        circuit.append(cirq.rz(parameters[parameter_counter + 14]).on(qubit1))
        parameter_counter += 15

    circuit.append(cirq.rz(parameters[parameter_counter]).on(qubits[0]))
    circuit.append(cirq.ry(parameters[parameter_counter + 1]).on(qubits[0]))
    circuit.append(cirq.rz(parameters[parameter_counter + 2]).on(qubits[0]))
    return circuit


def get_parameters_for_MPS_layer(mps):
    mps = copy.deepcopy(mps)
    number_of_qubits = len(mps)
    parameters = []

    for site in mps[1:][::-1]:  # Reversing order of MPS and leaving out first site
        unitary = _get_unitary_form_of_mps_site(site)
        parameters += _get_kak_decomposition_parameters_for_unitary(unitary)

    # Inserting kak decomp on first qubit to rotate first qubit respective to first tensor site
    unitary = _get_unitary_form_of_mps_site(mps[0])
    parameters += _get_kak_decomposition_parameters_for_unitary(unitary)
    parameters = [parameter % (2 * np.pi) for parameter in parameters]
    return parameters


def encode_bond_dimension_two_mps_as_quantum_circuit(mps):
    number_of_qubits = len(mps)
    parameters = get_parameters_for_MPS_layer(mps)

    qubits = [cirq.LineQubit(i) for i in range(number_of_qubits)]
    circuit = cirq.Circuit()
    circuit = add_MPS_layer(circuit, qubits, parameters)

    return circuit, qubits


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
