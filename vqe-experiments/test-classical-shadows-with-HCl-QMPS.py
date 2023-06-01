from openfermion import QubitOperator, QubitDavidson, count_qubits
import numpy as np
from qcmps.encoding.mps_encoding import add_MPS_layer
from qcmps.mps.mps import get_mps, get_truncated_mps
from qcmps.encoding.mps_encoding import get_parameters_for_MPS_layer
import cirq
import matplotlib.pyplot as plt
from classical_shadows import (
    estimate_sum_of_operators,
    sample_classical_shadow,
    _convert_qubit_operator_to_pauli_string_lists,
)


def get_hcl_hamiltonian():

    return (
        QubitOperator("", -438.638)
        + QubitOperator("Z2", 6.451)
        + QubitOperator("Z1", 6.588)
        + QubitOperator("Z1 Z2", 1.012)
        + QubitOperator("Z0", 6.545)
        + QubitOperator("Z0 Z2", 0.969)
        + QubitOperator("Z0 Z1", 1.106)
        + QubitOperator("X0 Z1 X2", -0.004)
        + QubitOperator("Y0 Z1 Y2", -0.004)
    )


def generate_QMPS_circuit(number_of_qubits, parameters):
    # Generate parameterized QMPS circuit
    # return parameterized_circuit
    number_of_parameters_per_layer = (15 * (number_of_qubits - 1)) + 3
    assert len(parameters) % number_of_parameters_per_layer == 0
    number_of_layers = int(len(parameters) / number_of_parameters_per_layer)
    qubits = [cirq.LineQubit(i) for i in range(number_of_qubits)]
    circuit = cirq.Circuit()
    for layer_index in range(number_of_layers):

        layer_parameters = parameters[
            layer_index
            * number_of_parameters_per_layer : (layer_index + 1)
            * number_of_parameters_per_layer
        ]
        circuit = add_MPS_layer(circuit, qubits, layer_parameters)
    return circuit


def get_state(parameters, number_of_qubits=12):
    simulator = cirq.Simulator()
    circuit = generate_QMPS_circuit(number_of_qubits, parameters)
    return np.asarray(simulator.simulate(circuit).final_state_vector)


def generate_cost_function(hamiltonian, cost_history=[]):
    simulator = cirq.Simulator()
    number_of_qubits = count_qubits(hamiltonian)
    terms, coefficients = _convert_qubit_operator_to_pauli_string_lists(hamiltonian)

    def cost_function(parameters):
        circuit = generate_QMPS_circuit(number_of_qubits, parameters)
        # wavefunction = np.asarray(simulator.simulate(circuit).final_state_vector)
        # cost = expectation(get_sparse_operator(hamiltonian), wavefunction)
        num_shots = 1000
        num_snapshots = 10000
        shadow_data = sample_classical_shadow(circuit, num_shots, num_snapshots)
        cost = estimate_sum_of_operators(shadow_data, terms, coefficients)
        cost_history.append(cost)
        return cost.real

    return cost_function


FCI_energy = -455.1570662167425
print("---- {} ---- FCI Energy: {}".format("HCl", FCI_energy))
hamiltonian = get_hcl_hamiltonian()
number_of_qubits = count_qubits(hamiltonian)
_, ground_state_energy, ground_state = QubitDavidson(
    hamiltonian, count_qubits(hamiltonian), options=None
).get_lowest_n(1)
ground_state = ground_state.reshape(2**number_of_qubits)
print("    Diagonalized Hamiltonian")
print(
    "        Diagonalized Ground State Energy: {} Error: {}".format(
        ground_state_energy[0], ground_state_energy[0] - FCI_energy
    )
)

number_of_layers = 1
cost_function = generate_cost_function(hamiltonian, cost_history=[])

mps = get_mps(ground_state)
mps = get_truncated_mps(mps, 2)
mps_parameters = get_parameters_for_MPS_layer(mps)
print(
    "        Energy from Initialization: {} Error: {}".format(
        cost_function(mps_parameters),
        cost_function(mps_parameters) - FCI_energy,
    )
)
