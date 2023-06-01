import cirq
import numpy as np
from openfermion import QubitOperator, count_qubits


def generate_random_pauli_gates(qubits):
    pauli_gates = ["i", "x", "y", "z"]
    random_gate_sequence = "".join(np.random.choice(pauli_gates, qubits))
    return random_gate_sequence


def apply_pauli_gate(circuit, gate, qubit):
    if gate == "x":
        circuit.append(cirq.X(qubit))
    elif gate == "y":
        circuit.append(cirq.Y(qubit))
    elif gate == "z":
        circuit.append(cirq.Z(qubit))


def create_classical_shadow_circuit(state_circuit, qubits, pauli_gate_sequence):
    shadow_circuit = state_circuit.copy()
    for i, gate in enumerate(pauli_gate_sequence):
        apply_pauli_gate(shadow_circuit, gate, qubits[i])
    shadow_circuit.append(cirq.measure(*qubits, key="result"))
    return shadow_circuit


def sample_classical_shadow(state_circuit, num_shots, num_snapshots):
    n_qubits = len(state_circuit.all_qubits())
    qubits = sorted(list(state_circuit.all_qubits()))
    simulator = cirq.Simulator()
    shadow_data = []
    for _ in range(num_snapshots):
        pauli_gate_sequence = generate_random_pauli_gates(n_qubits)
        shadow_circuit = create_classical_shadow_circuit(
            state_circuit, qubits, pauli_gate_sequence
        )
        # state_vector = simulator.simulate(shadow_circuit).final_state_vector
        # probability_distribution = np.dot(state_vector.T.conj(), state_vector)
        # samples = np.random.choice(range(0, 2**n_qubits))
        result = simulator.run(shadow_circuit, repetitions=num_shots)
        outcomes = result.histogram(key="result")
        for outcome, count in outcomes.items():
            shadow_data.append(
                {
                    "outcome": outcome,
                    "count": count,
                    "gate_sequence": pauli_gate_sequence,
                }
            )
    return shadow_data


def estimate_expectation_value(shadow_data, observable):
    total_count = 0
    total_sum = 0
    for snapshot in shadow_data:
        outcome = snapshot["outcome"]
        count = snapshot["count"]
        gate_sequence = snapshot["gate_sequence"]
        sign = 1
        for qubit, gate in enumerate(gate_sequence):
            if gate != "i" and observable[qubit] == gate:
                sign = -sign
        total_sum += sign * count
        total_count += count
    return total_sum / total_count


def _convert_qubit_operator_to_pauli_string_lists(qubit_operator):
    number_of_qubits = count_qubits(qubit_operator)
    terms, coefficients = [], []
    for op in qubit_operator.get_operators():
        term = None
        for operator_term in op.terms:
            term = operator_term

        term_string = ""
        for i in range(number_of_qubits):
            found_term = False
            for pauli in term:
                if pauli[0] == i:
                    term_string += pauli[1].lower()
                    found_term = True
            if not found_term:
                term_string += "i"

        coefficient = qubit_operator.terms[term]
        terms.append(term_string)
        coefficients.append(coefficient)
    return terms, coefficients


def estimate_sum_of_operators(shadow, observables, coefficients):
    sum = 0
    for observable, coefficient in zip(observables, coefficients):
        sum += coefficient * estimate_expectation_value(shadow, observable)
    return sum


# # Example usage:
# num_qubits = 2
# qubits = [cirq.GridQubit(i, 0) for i in range(num_qubits)]
# state_circuit = cirq.Circuit()
# state_circuit.append(cirq.H(qubits[0]))
# state_circuit.append(cirq.CNOT(qubits[1], qubits[0]))
# num_shots = 100
# num_snapshots = 10
# shadow_data = sample_classical_shadow(state_circuit, num_shots, num_snapshots)
# hamiltonian = (
#     QubitOperator("", -438.638)
#     + QubitOperator("Z0", 6.451)
#     + QubitOperator("Z1", 6.588)
#     + QubitOperator("Z0 Z1", 1.106)
# )
# zero_state = np.zeros(2**num_qubits)
# zero_state[0] = 1
# state = cirq.unitary(state_circuit) @ zero_state
# from openfermion.linalg import expectation, get_sparse_operator

# print(expectation(get_sparse_operator(hamiltonian), state))
# terms, coefficients = _convert_qubit_operator_to_pauli_string_lists(hamiltonian)
# print(estimate_sum_of_operators(shadow_data, terms, coefficients))
# print(shadow_data)

