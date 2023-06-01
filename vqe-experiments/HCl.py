from openfermion.chem import MolecularData
from openfermion import QubitOperator, QubitDavidson, count_qubits, get_sparse_operator
from openfermion.linalg import expectation
from symmer.operators import PauliwordOp, QuantumState
import numpy as np
from symmer.projection import QubitTapering, ContextualSubspace
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


def get_hcl_molecule(bond_length=1.341):
    return MolecularData(
        geometry=[
            ["H", [0.0, 0.0, 0.0]],
            ["Cl", [0.0, 0.0, bond_length]],
        ],
        basis="sto-3g",
        multiplicity=1,
        charge=0,
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


def run_vqe(hamiltonian, number_of_qubits, initial_parameters):
    from scipy.optimize import minimize

    cost_history = []
    cost_function = generate_cost_function(hamiltonian, cost_history=cost_history)
    results = minimize(
        cost_function,
        initial_parameters,
        method="Nelder-Mead",
        tol=1e-12,
        options={"maxfev": 10000},
    )
    opt_value = cost_function(results.x)
    return (
        opt_value,
        results.x,
        get_state(results.x, number_of_qubits=number_of_qubits),
        cost_history,
    )


def run_and_plot(molecule):
    FCI_energy = -455.1570662167425
    print("---- {} ---- FCI Energy: {}".format(molecule.name, FCI_energy))
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

    # initial_parameters = np.random.uniform(
    #     0, 2 * np.pi, number_of_layers * ((15 * (number_of_qubits - 1)) + 3)
    # ).tolist()
    # _, parameters, _, random_cost_history = run_vqe(
    #     hamiltonian, number_of_qubits, initial_parameters
    # )
    # print("    Finished Random Initialization")
    # print(
    #     "        Energy from Optimization: {} {} Error: {}".format(
    #         random_cost_history[-1],
    #         cost_function(parameters),
    #         cost_function(parameters) - FCI_energy,
    #     )
    # )

    # random_error_history = [abs(energy - FCI_energy) for energy in random_cost_history]

    mps = get_mps(ground_state)
    mps = get_truncated_mps(mps, 2)
    mps_parameters = get_parameters_for_MPS_layer(mps)
    print(
        "        Energy from Initialization: {} Error: {}".format(
            cost_function(mps_parameters),
            cost_function(mps_parameters) - FCI_energy,
        )
    )
    # _, parameters, _, mps_cost_history = run_vqe(
    #     hamiltonian, number_of_qubits, mps_parameters
    # )
    # mps_error_history = [abs(energy - FCI_energy) for energy in mps_cost_history]
    # print("    Finished MPS Initialization")
    # print(
    #     "        Energy from Initialization: {} Error: {}".format(
    #         mps_cost_history[0],
    #         mps_cost_history[0] - FCI_energy,
    #     )
    # )
    # print(
    #     "        Energy from Optimization: {} {} Error: {}".format(
    #         mps_cost_history[-1],
    #         cost_function(parameters),
    #         cost_function(parameters) - FCI_energy,
    #     )
    # )

    # plt.rc("font", size=10)  # controls default text size
    # plt.rc("axes", titlesize=10)  # fontsize of the title
    # plt.rc("axes", labelsize=10)  # fontsize of the x and y labels
    # plt.rc("xtick", labelsize=10)  # fontsize of the x tick labels
    # plt.rc("ytick", labelsize=10)  # fontsize of the y tick labels
    # plt.rc("legend", fontsize=6)  # fontsize of the legend

    # length = 8 / 2.54  # convert inches to cm
    # width = 8 / 2.54  # convert inches to cm
    # fig = plt.figure(figsize=(width, length), tight_layout=True)
    # energy_ax = plt.subplot(111)
    # energy_color = "#4668ea"
    # energy_ax.plot(
    #     range(len(random_error_history)),
    #     random_error_history,
    #     label="Random",
    #     color=energy_color,
    #     lw=0.5,
    # )
    # energy_ax.plot(
    #     range(len(mps_error_history)),
    #     mps_error_history,
    #     label=r"$MPS_{\chi=2}$",
    #     color="#ef545a",
    #     lw=0.5,
    # )
    # plt.hlines(
    #     ground_state_energy - FCI_energy,
    #     0,
    #     max(len(mps_error_history), len(random_error_history)),
    #     color="black",
    #     ls="dashed",
    # )
    # # energy_ax.legend()
    # energy_ax.set_xlabel("Evaluation")
    # energy_ax.set_yscale("log")
    # energy_ax.set_ylabel("Energy Error (Ha)", color=energy_color)
    # energy_ax.set_ylim(0.02, 1e1)
    # plt.legend(loc="best")

    # for t in energy_ax.get_yticklabels():
    #     t.set_color(energy_color)

    # plt.savefig(
    #     "qmps_{}.pdf".format(molecule.name),
    #     dpi=300,
    # )
    # plt.close()


molecule = get_hcl_molecule()
run_and_plot(molecule)
# hamiltonian = get_hcl_hamiltonian()
# _, ground_state_energy, ground_state = QubitDavidson(
#     hamiltonian, count_qubits(hamiltonian), options=None
# ).get_lowest_n(1)
# print(ground_state.round(2))
# print(ground_state_energy)
