from openfermion import QubitOperator, get_sparse_operator
import cirq
import numpy as np
from openfermion.linalg import eigenspectrum, expectation
import matplotlib.pyplot as plt

from openfermion.chem import MolecularData
from openfermionpsi4 import run_psi4
from openfermion.transforms import get_fermion_operator, bravyi_kitaev, jordan_wigner
from openfermion import count_qubits
from src.encoding.mps_encoding import add_MPS_layer


def get_qubit_hamiltonian_from_molecule(molecule):
    try:
        molecule.load()
    except:
        molecule = run_psi4(
            molecule,
            run_scf=1,
            run_mp2=1,
            run_ccsd=1,
            run_cisd=1,
            run_fci=1,
            verbose=False,
        )
        molecule.save()
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
    number_of_layers = int(len(parameters) / number_of_parameters_per_layer)
    qubits = [cirq.LineQubit(i) for i in range(number_of_qubits)]
    circuit = cirq.Circuit()
    for layer_index in range(number_of_layers):
        layer_parameters = parameters[
            layer_index : (layer_index + 1) * number_of_parameters_per_layer
        ]
        circuit = add_MPS_layer(circuit, qubits, layer_parameters)
    return circuit


def generate_cost_function(hamiltonian):
    simulator = cirq.Simulator()
    number_of_qubits = count_qubits(hamiltonian)

    def cost_function(parameters):
        circuit = generate_QMPS_circuit(number_of_qubits, parameters)
        wavefunction = np.asarray(simulator.simulate(circuit).final_state_vector)
        import pdb

        pdb.set_trace()
        cost = expectation(get_sparse_operator(hamiltonian), wavefunction)
        return cost.real

    return cost_function


def run_vqe_optimization_on_molecule(molecule):
    from scipy.optimize import minimize

    initial_parameters = np.random.uniform(0, 2 * np.pi, 2).tolist()

    hamiltonian = get_qubit_hamiltonian_from_molecule(molecule)
    ground_state_energy = eigenspectrum(hamiltonian)[0]
    cost_function = generate_cost_function(hamiltonian)

    results = minimize(cost_function, initial_parameters, method="Nelder-Mead")

    opt_value = cost_function(results.x)
    print(
        "Energy is: {} | Error is: {}".format(
            opt_value, abs(opt_value - ground_state_energy)
        )
    )
    return opt_value, results.x


molecule = MolecularData(
    geometry=[
        ["Li", [0.0, 0.0, 0.0]],
        ["H", [0.0, 0.0, 1.595]],
    ],
    basis="sto-3g",
    multiplicity=1,
)
run_vqe_optimization_on_molecule(molecule)
