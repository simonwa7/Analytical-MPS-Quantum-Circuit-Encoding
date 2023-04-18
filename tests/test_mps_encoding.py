from src.encoding.mps_encoding import (
    is_unitary,
    get_unitary_form_of_mps_site,
    encode_bond_dimension_two_mps_as_quantum_circuit,
    encode_mps_in_quantum_circuit,
    add_MPS_layer,
)
import pytest
import numpy as np
import copy
from itertools import combinations
import math
import cirq
from src.mps.mps import get_random_mps, get_wavefunction, get_truncated_mps, get_mps

SEED = 1234
np.random.seed(SEED)


@pytest.mark.parametrize(
    "matrix",
    [
        np.eye(1),
        np.eye(2),
        np.eye(3),
        np.eye(4),
        np.asarray([[0, 1], [1, 0]]),
        np.asarray([[0, -1j], [1j, 0]]),
        np.asarray([[1, 0], [0, -1]]),
    ],
)
def test_matrix_is_unitary(matrix):
    assert is_unitary(matrix)


@pytest.mark.parametrize(
    "matrix",
    [
        np.zeros((1, 1)),
        np.zeros((2, 2)),
        np.zeros((3, 3)),
        np.zeros((4, 4)),
        (1 / np.sqrt(2)) * np.asarray([[1, 1], [1, 1]]),
        np.asarray([[0, 1j], [0, -1j]]),
    ],
)
def test_matrix_is_not_unitary(matrix):
    assert not is_unitary(matrix)


@pytest.mark.parametrize(
    "matrix",
    [
        np.eye(1),
        np.eye(2),
        np.eye(3),
        np.eye(4),
        np.asarray([[0, 1], [1, 0]]),
        np.asarray([[0, -1j], [1j, 0]]),
        np.asarray([[1, 0], [0, -1]]),
    ],
)
def test_matrix_is_unitary_doesnt_destroy_input(matrix):
    copied_matrix = copy.deepcopy(matrix)
    is_unitary(matrix)
    assert np.array_equal(copied_matrix, matrix)


@pytest.mark.parametrize("number_of_sites", range(3, 15))
def test_get_unitary_form_of_mps_site(number_of_sites):
    # Step 1: test that result is unitary
    # Step 2: test that matrix properly encodes mps site
    mps = get_random_mps(number_of_sites, 2, complex=True)
    for i, site in enumerate(mps):
        unitary = get_unitary_form_of_mps_site(site)
        assert is_unitary(unitary, atol=1e-12)
        assert len(unitary.shape) == 2
        assert unitary.shape[0] == unitary.shape[1]
        assert math.log(unitary.shape[0], 2).is_integer()


def test_get_unitary_form_of_mps_site_doesnt_destory_input():
    mps = get_random_mps(10, 2, complex=True)
    mps_site = mps[np.random.choice(range(10))]
    copied_mps_site = copy.deepcopy(mps_site)
    _ = get_unitary_form_of_mps_site(mps_site)
    assert np.array_equal(mps_site, copied_mps_site)


def test_get_unitary_form_of_mps_site_raises_error_on_not_bd2_site():
    mps = get_random_mps(4, 8, complex=True)
    mps_site = mps[1]
    pytest.raises(AssertionError, get_unitary_form_of_mps_site, mps_site)


@pytest.mark.parametrize("number_of_sites", range(1, 12))
def test_encode_bond_dimension_two_mps_as_quantum_circuit(number_of_sites):
    # Randomly generate a bunch of bond_dimension 2 MPS's
    # Generate + sim circuits
    # Assert that wf is consistent (up to global phase)
    mps = get_random_mps(number_of_sites, 2, complex=True)
    mps_wf = get_wavefunction(copy.deepcopy(mps))
    circuit, _ = encode_bond_dimension_two_mps_as_quantum_circuit(mps)

    zero_state = np.zeros(2 ** number_of_sites)
    zero_state[0] = 1
    prepared_state = cirq.unitary(circuit) @ zero_state
    overlap = abs(np.dot(mps_wf.reshape(2 ** len(mps)).T.conj(), prepared_state))
    assert overlap > (1 - 1e-15)


@pytest.mark.parametrize("number_of_sites", range(12, 22))
def test_encode_bond_dimension_two_mps_as_quantum_circuit_larger_systems(
    number_of_sites,
):
    # Randomly generate a bunch of bond_dimension 2 MPS's
    # Generate + sim circuits
    # Assert that wf is consistent (up to global phase)
    mps = get_random_mps(number_of_sites, 2, complex=True)
    mps_wf = get_wavefunction(copy.deepcopy(mps))
    circuit, _ = encode_bond_dimension_two_mps_as_quantum_circuit(mps)

    simulator = cirq.Simulator()
    prepared_state = np.asarray(simulator.simulate(circuit).final_state_vector)
    overlap = abs(np.dot(mps_wf.reshape(2 ** len(mps)).T.conj(), prepared_state))
    assert overlap > (1 - 1e-6)


@pytest.mark.parametrize("sites", combinations([0, 1, 2, 3], 2))
def test_encode_bond_dimension_two_mps_as_quantum_circuit_bell_state(sites):
    bell_state_wf = np.zeros(4)
    bell_state_wf[sites[0]] = 1.0 / np.sqrt(2)
    bell_state_wf[sites[1]] = 1.0 / np.sqrt(2)
    u, s, v = np.linalg.svd(bell_state_wf.reshape(2, 2), full_matrices=False)
    bell_state_mps = [
        u.reshape(1, 2, 2),
        (np.diag(s) @ v).reshape(2, 2, 1),
    ]

    circuit, _ = encode_bond_dimension_two_mps_as_quantum_circuit(bell_state_mps)

    zero_state = np.zeros(4)
    zero_state[0] = 1
    prepared_state = cirq.unitary(circuit) @ zero_state
    overlap = abs(np.dot(bell_state_wf.T.conj(), prepared_state))

    assert overlap > (1 - 1e-15)


def test_encode_bond_dimension_two_mps_as_quantum_circuit_doesnt_destroy_input():
    mps = get_random_mps(10, 2, complex=True)
    copied_mps = copy.deepcopy(mps)
    _, _ = encode_bond_dimension_two_mps_as_quantum_circuit(mps)
    for site, copied_site in zip(mps, copied_mps):
        assert np.array_equal(site, copied_site)


def test_encode_bond_dimension_two_mps_as_quantum_circuit_raises_assertion_when_bond_dimension_is_not_2():
    mps = get_random_mps(10, 4, complex=True)
    pytest.raises(AssertionError, encode_bond_dimension_two_mps_as_quantum_circuit, mps)


# # @pytest.mark.parametrize("number_of_sites", range(2, 10))
# # def test_encode_mps_in_quantum_circuit_random_mps_gets_better_with_additional_layers(
# #     number_of_sites,
# # ):
# #     # Randomly generate a bunch of arbitrary bd MPS's
# #     # Generate + sim circuits
# #     # Assert that wf is consistent (up to global phase)
# #     mps = get_random_mps(number_of_sites, 2 ** int(number_of_sites / 2), complex=True)
# #     mps_wf = get_wavefunction(copy.deepcopy(mps))
# #     max_bond_dimension = max([site.shape[2] for site in mps])

# #     previous_error = np.inf
# #     for number_of_layers in range(int(max_bond_dimension / 2)):
# #         circuit, _, _ = encode_mps_in_quantum_circuit(
# #             mps, number_of_layers=number_of_layers
# #         )

# #         simulator = cirq.Simulator()
# #         result = simulator.simulate(circuit)
# #         qc_wf = result.final_state_vector

# #         current_error = np.linalg.norm(abs(mps_wf ** 2) - abs(qc_wf ** 2))
# #         assert current_error <= previous_error
# #         previous_error = current_error

# #     # for mps_prob, circuit_prob in zip(abs(mps_wf ** 2), abs(qc_wf ** 2)):
# #     #     assert np.isclose(mps_prob, circuit_prob, atol=1e-2, rtol=1e-2)
# #     # np.testing.assert_allclose(
# #     #     abs(mps_wf ** 2),
# #     #     abs(qc_wf ** 2),
# #     #     atol=1e-6,
# #     # )
# #     # cirq.testing.assert_allclose_up_to_global_phase(mps_wf, qc_wf, rtol=1e-1, atol=1e-1)


# @pytest.mark.parametrize("number_of_sites", [2, 3])
# def test_encode_mps_in_quantum_circuit_maximal_bell_state_for_two_and_three_sites_gives_near_perfect_overlap(number_of_sites):
#     bell_state_wf = np.zeros(2 ** number_of_sites)
#     bell_state_wf[0] = 1.0 / np.sqrt(2)
#     bell_state_wf[-1] = 1.0 / np.sqrt(2)
#     bell_state_mps = get_mps(bell_state_wf)

#     circuit, _ = encode_mps_in_quantum_circuit(bell_state_mps, number_of_layers=1)

#     simulator = cirq.Simulator()
#     result = simulator.simulate(circuit)
#     qc_wf = result.final_state_vector


# @pytest.mark.parametrize("number_of_sites", range(2, 12))
# def test_encode_mps_in_quantum_circuit_maximal_bell_state_gives_near_perfect_overlap(number_of_sites):
#     bell_state_wf = np.zeros(2 ** number_of_sites)
#     bell_state_wf[0] = 1.0 / np.sqrt(2)
#     bell_state_wf[-1] = 1.0 / np.sqrt(2)
#     bell_state_mps = get_mps(bell_state_wf)

#     circuit, _ = encode_mps_in_quantum_circuit(bell_state_mps)

#     simulator = cirq.Simulator()
#     result = simulator.simulate(circuit)
#     qc_wf = result.final_state_vector

#     np.testing.assert_allclose(
#         abs(bell_state_wf ** 2),
#         abs(qc_wf ** 2),
#         atol=1e-5,
#     )
#     cirq.testing.assert_allclose_up_to_global_phase(
#         bell_state_wf, qc_wf, rtol=1e-2, atol=1e-2
#     )


# def test_encode_mps_in_quantum_circuit_doesnt_destroy_input():
#     mps = get_random_mps(10, 1024, complex=True)
#     copied_mps = copy.deepcopy(mps)
#     _, _, _ = encode_mps_in_quantum_circuit(mps)
#     for site, copied_site in zip(mps, copied_mps):
#         assert np.array_equal(site, copied_site)


def test_add_MPS_layer_does_not_alter_input_parameters():
    number_of_qubits = 5
    qubits = [cirq.LineQubit(i) for i in range(number_of_qubits)]
    circuit = cirq.Circuit()
    parameters = np.random.uniform(0, 2 * np.pi, 63)
    copied_parameters = copy.deepcopy(parameters)
    add_MPS_layer(circuit, qubits, parameters)
    np.testing.assert_array_equal(parameters, copied_parameters)


@pytest.mark.parametrize("number_of_parameters", [0, 1, 10, 17, 40, 62, 64, 78, 63 * 2])
def test_add_MPS_layer_checks_number_of_parameters_is_correct(number_of_parameters):
    number_of_qubits = 5
    qubits = [cirq.LineQubit(i) for i in range(number_of_qubits)]
    circuit = cirq.Circuit()
    parameters = np.random.uniform(0, 2 * np.pi, number_of_parameters)

    pytest.raises(AssertionError, add_MPS_layer, circuit, qubits, parameters)


def test_add_MPS_layer_adds_to_existing_circuit():
    number_of_qubits = 5
    qubits = [cirq.LineQubit(i) for i in range(number_of_qubits)]
    circuit = cirq.Circuit()
    circuit.append(cirq.X.on(qubits[0]))
    parameters = np.random.uniform(0, 2 * np.pi, 63)

    circuit = add_MPS_layer(circuit, qubits, parameters)
    assert isinstance(
        circuit.operation_at(qubits[0], 0).gate, cirq.ops.pauli_gates._PauliX
    )


def test_add_MPS_layer_adds_parameters_in_correct_order_one_qubit():
    number_of_qubits = 1
    number_of_parameters = 3
    qubits = [cirq.LineQubit(i) for i in range(number_of_qubits)]
    circuit = cirq.Circuit()
    parameters = np.random.uniform(0, 2 * np.pi, number_of_parameters)

    circuit = add_MPS_layer(circuit, qubits, parameters)
    assert circuit.operation_at(qubits[0], 0).gate._rads == parameters[0]
    assert circuit.operation_at(qubits[0], 1).gate._rads == parameters[1]
    assert circuit.operation_at(qubits[0], 2).gate._rads == parameters[2]


@pytest.mark.parametrize("number_of_qubits", range(2, 12))
def test_add_MPS_layer_adds_parameters_in_correct_order(number_of_qubits):
    number_of_parameters = (15 * (number_of_qubits - 1)) + 3
    qubits = [cirq.LineQubit(i) for i in range(number_of_qubits)]
    circuit = cirq.Circuit()
    parameters = np.random.uniform(0, 2 * np.pi, number_of_parameters)

    circuit = add_MPS_layer(circuit, qubits, parameters)

    # Check first KAK decomp
    assert circuit.operation_at(qubits[-2], 0).gate._rads == parameters[0]
    assert circuit.operation_at(qubits[-2], 1).gate._rads == parameters[1]
    assert circuit.operation_at(qubits[-2], 2).gate._rads == parameters[2]

    assert circuit.operation_at(qubits[-1], 0).gate._rads == parameters[3]
    assert circuit.operation_at(qubits[-1], 1).gate._rads == parameters[4]
    assert circuit.operation_at(qubits[-1], 2).gate._rads == parameters[5]

    np.testing.assert_almost_equal(
        circuit.operation_at(qubits[-2], 3).gate.exponent,
        (-2 / np.pi) * parameters[6],
        12,
    )
    np.testing.assert_almost_equal(
        circuit.operation_at(qubits[-2], 4).gate.exponent,
        (-2 / np.pi) * parameters[7],
        12,
    )
    np.testing.assert_almost_equal(
        circuit.operation_at(qubits[-2], 5).gate.exponent,
        (-2 / np.pi) * parameters[8],
        12,
    )

    assert circuit.operation_at(qubits[-2], 6).gate._rads == parameters[9]
    assert circuit.operation_at(qubits[-2], 7).gate._rads == parameters[10]
    assert circuit.operation_at(qubits[-2], 8).gate._rads == parameters[11]

    assert circuit.operation_at(qubits[-1], 6).gate._rads == parameters[12]
    assert circuit.operation_at(qubits[-1], 7).gate._rads == parameters[13]
    assert circuit.operation_at(qubits[-1], 8).gate._rads == parameters[14]

    if number_of_qubits > 2:
        # Check second KAK decomp
        assert circuit.operation_at(qubits[-3], 0).gate._rads == parameters[15]
        assert circuit.operation_at(qubits[-3], 1).gate._rads == parameters[16]
        assert circuit.operation_at(qubits[-3], 2).gate._rads == parameters[17]

        assert circuit.operation_at(qubits[-2], 9).gate._rads == parameters[18]
        assert circuit.operation_at(qubits[-2], 10).gate._rads == parameters[19]
        assert circuit.operation_at(qubits[-2], 11).gate._rads == parameters[20]

        np.testing.assert_almost_equal(
            circuit.operation_at(qubits[-3], 12).gate.exponent,
            (-2 / np.pi) * parameters[21],
            12,
        )
        np.testing.assert_almost_equal(
            circuit.operation_at(qubits[-3], 13).gate.exponent,
            (-2 / np.pi) * parameters[22],
            12,
        )
        np.testing.assert_almost_equal(
            circuit.operation_at(qubits[-3], 14).gate.exponent,
            (-2 / np.pi) * parameters[23],
            12,
        )

        assert circuit.operation_at(qubits[-3], 15).gate._rads == parameters[24]
        assert circuit.operation_at(qubits[-3], 16).gate._rads == parameters[25]
        assert circuit.operation_at(qubits[-3], 17).gate._rads == parameters[26]

        assert circuit.operation_at(qubits[-2], 15).gate._rads == parameters[27]
        assert circuit.operation_at(qubits[-2], 16).gate._rads == parameters[28]
        assert circuit.operation_at(qubits[-2], 17).gate._rads == parameters[29]

    if number_of_qubits > 2:
        # Check last KAK decomp (second last unitary)
        assert circuit.moments[0].operations[-1].gate._rads == parameters[-18]
        assert circuit.moments[1].operations[-1].gate._rads == parameters[-17]
        assert circuit.moments[2].operations[-1].gate._rads == parameters[-16]

        assert circuit.moments[-12].operations[0].gate._rads == parameters[-15]
        assert circuit.moments[-11].operations[0].gate._rads == parameters[-14]
        assert circuit.moments[-10].operations[0].gate._rads == parameters[-13]

        np.testing.assert_almost_equal(
            circuit.moments[-9].operations[0].gate.exponent,
            (-2 / np.pi) * parameters[-12],
            12,
        )
        np.testing.assert_almost_equal(
            circuit.moments[-8].operations[0].gate.exponent,
            (-2 / np.pi) * parameters[-11],
            12,
        )
        np.testing.assert_almost_equal(
            circuit.moments[-7].operations[0].gate.exponent,
            (-2 / np.pi) * parameters[-10],
            12,
        )

        assert circuit.moments[-6].operations[0].gate._rads == parameters[-9]
        assert circuit.moments[-5].operations[0].gate._rads == parameters[-8]
        assert circuit.moments[-4].operations[0].gate._rads == parameters[-7]

        assert circuit.moments[-6].operations[1].gate._rads == parameters[-6]
        assert circuit.moments[-5].operations[1].gate._rads == parameters[-5]
        assert circuit.moments[-4].operations[1].gate._rads == parameters[-4]

    # Check final unitary
    assert circuit.moments[-3].operations[0].gate._rads == parameters[-3]
    assert circuit.moments[-2].operations[0].gate._rads == parameters[-2]
    assert circuit.moments[-1].operations[0].gate._rads == parameters[-1]


def test_add_MPS_layer_outputs_pqc_with_correct_structure():
    number_of_qubits = 4
    number_of_parameters = (15 * (number_of_qubits - 1)) + 3
    qubits = [cirq.LineQubit(i) for i in range(number_of_qubits)]
    circuit = cirq.Circuit()
    parameters = np.random.uniform(0, 2 * np.pi, number_of_parameters)

    circuit = add_MPS_layer(circuit, qubits, parameters)

    # Check first KAK decomp on qubits 2&3
    assert isinstance(circuit.operation_at(qubits[3], 0).gate, cirq.ops.common_gates.Rz)
    assert isinstance(circuit.operation_at(qubits[3], 1).gate, cirq.ops.common_gates.Ry)
    assert isinstance(circuit.operation_at(qubits[3], 2).gate, cirq.ops.common_gates.Rz)
    assert isinstance(circuit.operation_at(qubits[2], 0).gate, cirq.ops.common_gates.Rz)
    assert isinstance(circuit.operation_at(qubits[2], 1).gate, cirq.ops.common_gates.Ry)
    assert isinstance(circuit.operation_at(qubits[2], 2).gate, cirq.ops.common_gates.Rz)
    assert isinstance(
        circuit.operation_at(qubits[2], 3).gate, cirq.ops.parity_gates.XXPowGate
    )
    assert isinstance(
        circuit.operation_at(qubits[2], 4).gate, cirq.ops.parity_gates.YYPowGate
    )
    assert isinstance(
        circuit.operation_at(qubits[2], 5).gate, cirq.ops.parity_gates.ZZPowGate
    )
    assert isinstance(circuit.operation_at(qubits[3], 6).gate, cirq.ops.common_gates.Rz)
    assert isinstance(circuit.operation_at(qubits[3], 7).gate, cirq.ops.common_gates.Ry)
    assert isinstance(circuit.operation_at(qubits[3], 8).gate, cirq.ops.common_gates.Rz)
    assert isinstance(circuit.operation_at(qubits[2], 6).gate, cirq.ops.common_gates.Rz)
    assert isinstance(circuit.operation_at(qubits[2], 7).gate, cirq.ops.common_gates.Ry)
    assert isinstance(circuit.operation_at(qubits[2], 8).gate, cirq.ops.common_gates.Rz)

    # Check second KAK decomp on qubits 1&2
    assert isinstance(circuit.operation_at(qubits[2], 9).gate, cirq.ops.common_gates.Rz)
    assert isinstance(
        circuit.operation_at(qubits[2], 10).gate, cirq.ops.common_gates.Ry
    )
    assert isinstance(
        circuit.operation_at(qubits[2], 11).gate, cirq.ops.common_gates.Rz
    )
    assert isinstance(circuit.operation_at(qubits[1], 0).gate, cirq.ops.common_gates.Rz)
    assert isinstance(circuit.operation_at(qubits[1], 1).gate, cirq.ops.common_gates.Ry)
    assert isinstance(circuit.operation_at(qubits[1], 2).gate, cirq.ops.common_gates.Rz)
    assert isinstance(
        circuit.operation_at(qubits[1], 12).gate, cirq.ops.parity_gates.XXPowGate
    )
    assert isinstance(
        circuit.operation_at(qubits[1], 13).gate, cirq.ops.parity_gates.YYPowGate
    )
    assert isinstance(
        circuit.operation_at(qubits[1], 14).gate, cirq.ops.parity_gates.ZZPowGate
    )
    assert isinstance(
        circuit.operation_at(qubits[2], 15).gate, cirq.ops.common_gates.Rz
    )
    assert isinstance(
        circuit.operation_at(qubits[2], 16).gate, cirq.ops.common_gates.Ry
    )
    assert isinstance(
        circuit.operation_at(qubits[2], 17).gate, cirq.ops.common_gates.Rz
    )
    assert isinstance(
        circuit.operation_at(qubits[1], 15).gate, cirq.ops.common_gates.Rz
    )
    assert isinstance(
        circuit.operation_at(qubits[1], 16).gate, cirq.ops.common_gates.Ry
    )
    assert isinstance(
        circuit.operation_at(qubits[1], 17).gate, cirq.ops.common_gates.Rz
    )

    # Check third KAK decomp on qubits 0&1
    assert isinstance(
        circuit.operation_at(qubits[1], 18).gate, cirq.ops.common_gates.Rz
    )
    assert isinstance(
        circuit.operation_at(qubits[1], 19).gate, cirq.ops.common_gates.Ry
    )
    assert isinstance(
        circuit.operation_at(qubits[1], 20).gate, cirq.ops.common_gates.Rz
    )
    assert isinstance(circuit.operation_at(qubits[0], 0).gate, cirq.ops.common_gates.Rz)
    assert isinstance(circuit.operation_at(qubits[0], 1).gate, cirq.ops.common_gates.Ry)
    assert isinstance(circuit.operation_at(qubits[0], 2).gate, cirq.ops.common_gates.Rz)
    assert isinstance(
        circuit.operation_at(qubits[0], 21).gate, cirq.ops.parity_gates.XXPowGate
    )
    assert isinstance(
        circuit.operation_at(qubits[0], 22).gate, cirq.ops.parity_gates.YYPowGate
    )
    assert isinstance(
        circuit.operation_at(qubits[0], 23).gate, cirq.ops.parity_gates.ZZPowGate
    )
    assert isinstance(
        circuit.operation_at(qubits[1], 24).gate, cirq.ops.common_gates.Rz
    )
    assert isinstance(
        circuit.operation_at(qubits[1], 25).gate, cirq.ops.common_gates.Ry
    )
    assert isinstance(
        circuit.operation_at(qubits[1], 26).gate, cirq.ops.common_gates.Rz
    )
    assert isinstance(
        circuit.operation_at(qubits[0], 24).gate, cirq.ops.common_gates.Rz
    )
    assert isinstance(
        circuit.operation_at(qubits[0], 25).gate, cirq.ops.common_gates.Ry
    )
    assert isinstance(
        circuit.operation_at(qubits[0], 26).gate, cirq.ops.common_gates.Rz
    )

    # Check final unitary on qubit 0
    assert isinstance(
        circuit.operation_at(qubits[0], 27).gate, cirq.ops.common_gates.Rz
    )
    assert isinstance(
        circuit.operation_at(qubits[0], 28).gate, cirq.ops.common_gates.Ry
    )
    assert isinstance(
        circuit.operation_at(qubits[0], 29).gate, cirq.ops.common_gates.Rz
    )
