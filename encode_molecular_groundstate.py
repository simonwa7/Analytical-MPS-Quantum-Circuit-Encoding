from openfermion.chem import MolecularData
from openfermionpsi4 import run_psi4
from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermion import count_qubits, QubitDavidson, get_sparse_operator, expectation
import numpy as np
from qcmps.encoding.mps_encoding import encode_mps_in_quantum_circuit


def get_qubit_hamiltonian_from_molecule(molecule):
    molecule = run_psi4(
        molecule,
        run_fci=1,
        verbose=True,
    )
    molecule.load()
    # Get the Hamiltonian in an active space.
    molecular_hamiltonian = molecule.get_molecular_hamiltonian()

    # Map operator to fermions and qubits.
    fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)
    qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)
    qubit_hamiltonian.compress()
    return qubit_hamiltonian


molecule = MolecularData(
    geometry=[
        ["Li", [0.0, 0.0, 0.0]],
        ["H", [0.0, 0.0, 1.565]],
    ],
    basis="sto-3g",
    multiplicity=1,
    charge=0,
)
hamiltonian = get_qubit_hamiltonian_from_molecule(molecule)
number_of_qubits = count_qubits(hamiltonian)
print("Got Hamiltonian")
_, ground_state_energy, ground_state = QubitDavidson(
    hamiltonian, count_qubits(hamiltonian), options=None
).get_lowest_n(1)
ground_state = ground_state.reshape(
    2**number_of_qubits,
)
print("Energy from Exact Diagonlization: ", ground_state_energy, flush=True)

from qcmps.mps.mps import get_mps, get_truncated_mps, get_wavefunction

mps = get_mps(ground_state)
print("Got MPS")
for bond_dimension in [2, 4, 8, 16, 32, 64]:
    print("---- Bond Dimension {} ----".format(bond_dimension))
    truncated_mps = get_truncated_mps(mps, bond_dimension)
    truncated_mps_wf = get_wavefunction(truncated_mps).reshape(2**number_of_qubits)
    truncation_overlap = abs(np.dot(ground_state.T.conj(), truncated_mps_wf))
    print("    Overlap with Truncated MPS: ", truncation_overlap)

    qmps, _, _ = encode_mps_in_quantum_circuit(truncated_mps)
    qmps_state = qmps.final_state_vector()

    overlap = abs(np.dot(ground_state.T.conj(), qmps_state))
    print(
        "    Energy from Quantum Circui with Bond Dimension {}: ".format(
            bond_dimension
        ),
        expectation(get_sparse_operator(hamiltonian), qmps_state).real,
    )
    print("    Overlap with QMPS: ", overlap)
