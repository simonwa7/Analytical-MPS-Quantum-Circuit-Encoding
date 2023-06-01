from openfermion import QubitOperator, get_sparse_operator
import cirq
import numpy as np
from openfermion.linalg import eigenspectrum, expectation
import matplotlib.pyplot as plt

from openfermion.chem import MolecularData
from openfermionpsi4 import run_psi4
from openfermion.transforms import get_fermion_operator, bravyi_kitaev, jordan_wigner
from openfermion import count_qubits, QubitDavidson
from qcmps.encoding.mps_encoding import add_MPS_layer


def get_qubit_hamiltonian_from_molecule(molecule):
    molecule = run_psi4(
        molecule,
        run_fci=1,
        verbose=False,
    )
    molecule.load()
    # Get the Hamiltonian in an active space.
    molecular_hamiltonian = molecule.get_molecular_hamiltonian()

    # Map operator to fermions and qubits.
    fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)
    qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)
    qubit_hamiltonian.compress()
    return qubit_hamiltonian


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


def generate_cost_function(hamiltonian):
    simulator = cirq.Simulator()
    number_of_qubits = count_qubits(hamiltonian)

    def cost_function(parameters):
        circuit = generate_QMPS_circuit(number_of_qubits, parameters)
        wavefunction = np.asarray(simulator.simulate(circuit).final_state_vector)
        cost = expectation(get_sparse_operator(hamiltonian), wavefunction)
        return cost.real

    return cost_function


def run_vqe(hamiltonian, number_of_qubits, number_of_layers):
    from scipy.optimize import minimize

    initial_parameters = np.random.uniform(
        0, 2 * np.pi, number_of_layers * ((15 * (number_of_qubits - 1)) + 3)
    ).tolist()

    cost_function = generate_cost_function(hamiltonian)
    results = minimize(cost_function, initial_parameters, method="Nelder-Mead")
    opt_value = cost_function(results.x)
    return opt_value, results.x, get_state(results.x, number_of_qubits=number_of_qubits)


molecule = MolecularData(
    geometry=[
        ["H", [0.0, 0.0, 0.0]],
        ["H", [0.0, 0.0, 0.7414]],
        ["H", [0.0, 0.0, 2 * 0.7414]],
    ],
    basis="sto-3g",
    multiplicity=1,
    charge=1,
)
hamiltonian = get_qubit_hamiltonian_from_molecule(molecule)
number_of_qubits = count_qubits(hamiltonian)
_, ground_state_energy, ground_state = QubitDavidson(
    hamiltonian, count_qubits(hamiltonian), options=None
).get_lowest_n(1)
ground_state = ground_state.reshape(2 ** number_of_qubits)

energies = []
errors = []
overlaps = []
infidelities = []
numbers_of_layers = range(1, 7)
for number_of_layers in numbers_of_layers:
    energy, _, state = run_vqe(hamiltonian, number_of_qubits, number_of_layers)
    overlap = abs(np.dot(ground_state.T.conj(), state))
    energies.append(energy)
    errors.append(abs(energy - ground_state_energy))
    overlaps.append(overlap)
    infidelities.append(1 - overlap ** 2)

plt.rc("font", size=10)  # controls default text size
plt.rc("axes", titlesize=10)  # fontsize of the title
plt.rc("axes", labelsize=10)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=10)  # fontsize of the x tick labels
plt.rc("ytick", labelsize=10)  # fontsize of the y tick labels
plt.rc("legend", fontsize=10)  # fontsize of the legend

length = 6 / 2.54  # convert inches to cm
width = 8 / 2.54  # convert inches to cm
fig = plt.figure(figsize=(width, length), tight_layout=True)
energy_ax = plt.subplot(111)
# energy_ax.set_title("Accuracy of H2 QMPS VQE")
energy_color = "#4668ea"
energy_ax.scatter(numbers_of_layers, errors, s=1, label="Random", color=energy_color)
energy_ax.plot(numbers_of_layers, errors, lw=1, label="Random", color=energy_color)
energy_ax.set_yscale("log")
plt.axhspan(1e-15, 0.00159, color="y", alpha=0.5, lw=0, label="Chemical Accuracy")
# energy_ax.legend()
energy_ax.set_xlabel("Number of Layers")
energy_ax.set_ylim(1e-12, 1e0)
energy_ax.set_ylabel("Energy Error (Ha)", color=energy_color)

infidelity_ax = energy_ax.twinx()
infidelity_color = "#ef545a"
infidelity_ax.scatter(
    numbers_of_layers,
    infidelities,
    s=1,
    color=infidelity_color,
    label="Infidelity",
)
infidelity_ax.plot(
    numbers_of_layers,
    infidelities,
    lw=1,
    ls="dashed",
    color=infidelity_color,
    label="Infidelity",
)
infidelity_ax.set_ylim(1e-12, 1e0)
infidelity_ax.set_yscale("log")
infidelity_ax.set_ylabel("Infidelity", color=infidelity_color)

for t in energy_ax.get_yticklabels():
    t.set_color(energy_color)

for t in infidelity_ax.get_yticklabels():
    t.set_color(infidelity_color)

plt.savefig(
    "error_and_infidelity_vs_number_of_layers_of_qmps_{}.pdf".format(molecule.name),
    dpi=300,
)
plt.close()
