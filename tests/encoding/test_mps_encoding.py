from qcmps.encoding.mps_encoding import (
    add_MPS_layer,
    get_parameters_for_MPS_layer,
    encode_bond_dimension_two_mps_as_quantum_circuit,
    encode_mps_in_quantum_circuit,
)
import pytest
import numpy as np
import copy
from itertools import combinations
import cirq
from qcmps.mps.mps import get_random_mps, get_wavefunction

SEED = 1234
np.random.seed(SEED)


def test_add_MPS_layer_does_not_alter_input_parameters():
    number_of_qubits = 5
    qubits = [cirq.LineQubit(i) for i in range(number_of_qubits)]
    circuit = cirq.Circuit()
    parameters = np.random.uniform(0, 2 * np.pi, 63)
    copied_parameters = copy.deepcopy(parameters)
    add_MPS_layer(circuit, qubits, parameters)
    assert np.array_equal(parameters, copied_parameters)


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


def test_get_parameters_for_MPS_layer_does_not_alter_inputs():
    mps = np.asarray(
        [
            np.asarray(
                [
                    [
                        [-0.57450887 + 0.0j, 0.81849836 + 0.0j],
                        [-0.75461618 + 0.31700785j, -0.52966959 + 0.2225097j],
                    ]
                ]
            ),
            np.asarray(
                [
                    [
                        [-0.80622012 - 0.15578776j, 0.07556346 + 0.3973943j],
                        [-0.5148668 + 0.00533767j, -0.14040741 - 0.75762631j],
                    ],
                    [
                        [-0.03357532 - 0.17394833j, -0.16929719 - 0.28482443j],
                        [0.02249959 - 0.16950346j, -0.36182116 - 0.04421552j],
                    ],
                ]
            ),
            np.asarray(
                [
                    [
                        [-0.69386917 - 2.40429501e-01j, 0.6462323 - 1.73402512e-01j],
                        [-0.38685763 - 5.46112579e-01j, -0.68057584 - 2.45821671e-01j],
                    ],
                    [
                        [-0.0465388 - 2.77407170e-02j, -0.15985938 - 1.45843588e-04j],
                        [-0.08004871 - 5.91459988e-02j, 0.01768885 - 5.32510290e-02j],
                    ],
                ]
            ),
            np.asarray(
                [
                    [
                        [-0.29989514 - 0.39514607j, 0.77801797 + 0.33581336j],
                        [-0.69129854 - 0.4785925j, -0.47930505 + 0.00670271j],
                    ],
                    [
                        [-0.12864227 - 0.14306525j, -0.03469456 - 0.15829426j],
                        [0.00403368 - 0.09972594j, 0.12353193 - 0.10304484j],
                    ],
                ]
            ),
            np.asarray(
                [
                    [
                        [-0.39729155 - 0.3216033j, -0.09427676 - 0.81537704j],
                        [-0.35509184 - 0.73184422j, -0.12365636 + 0.45901822j],
                    ],
                    [
                        [0.11244226 + 0.14638608j, 0.02484112 - 0.27863233j],
                        [0.13104029 + 0.16062646j, 0.11180199 + 0.09762586j],
                    ],
                ]
            ),
            np.asarray(
                [
                    [
                        [-0.46048036 - 0.37147234j, 0.11706143 - 0.79394302j],
                        [-0.64278269 - 0.47479187j, -0.06179588 + 0.57049114j],
                    ],
                    [
                        [-0.06854947 - 0.0594566j, -0.02852405 + 0.15190948j],
                        [-0.03180957 - 0.04607897j, 0.01479036 + 0.0506297j],
                    ],
                ]
            ),
            np.asarray(
                [
                    [
                        [-0.57424014 - 0.426199j, 0.54996946 - 0.41686379j],
                        [-0.50574004 - 0.48123054j, -0.6060921 + 0.36128866j],
                    ],
                    [
                        [-0.01908535 - 0.00975854j, -0.10919508 - 0.04882623j],
                        [-0.02113928 - 0.0184527j, -0.0610633 - 0.08857059j],
                    ],
                ]
            ),
            np.asarray(
                [
                    [
                        [-0.23988072 - 0.52156813j, 0.50736136 - 0.62579889j],
                        [-0.38884765 - 0.71307743j, -0.33559398 + 0.44770409j],
                    ],
                    [
                        [0.0437652 + 0.06077323j, 0.00668731 - 0.14697727j],
                        [0.02409024 + 0.06747517j, -0.10643432 - 0.07016141j],
                    ],
                ]
            ),
            np.asarray(
                [
                    [
                        [-0.755415 - 1.04083409e-17j, -0.6331034 + 1.16221928e-01j],
                        [-0.63744618 - 1.34084036e-01j, 0.75568693 + 1.98066558e-02j],
                    ],
                    [
                        [0.05726305 - 2.60208521e-18j, 0.02772025 + 1.34658635e-02j],
                        [0.03320878 + 2.54978495e-02j, 0.11003958 + 3.40448393e-02j],
                    ],
                ]
            ),
            np.asarray(
                [
                    [[0.66470115 + 0.0j], [0.66621201 - 0.26075466j]],
                    [[0.01717603 + 0.0j], [-0.01486055 + 0.0058164j]],
                ]
            ),
        ]
    )
    copied_mps = copy.deepcopy(mps)
    get_parameters_for_MPS_layer(mps)
    for site, copied_site in zip(mps, copied_mps):
        assert np.array_equal(site, copied_site)


@pytest.mark.parametrize(
    "mps",
    [
        np.asarray(
            [
                np.array(
                    [
                        [
                            [
                                -0.0510128971995174 + 0.7414751395178137j,
                                0.1792786734260809 + 0.6445707556590936j,
                            ],
                            [
                                -0.0435300869970359 + 0.6676207256537059j,
                                -0.2017219521991445 - 0.715329261392693j,
                            ],
                        ]
                    ]
                ),
                np.array(
                    [
                        [
                            [0.5667901365834367 + 0.0j],
                            [0.8113761461189671 - 0.1132975271123784j],
                        ],
                        [
                            [-0.0031232783596098 + 0.0j],
                            [0.0021400515819685 - 0.0002988287901853j],
                        ],
                    ]
                ),
            ]
        ),
        np.asarray(
            [
                np.array(
                    [
                        [
                            [
                                -0.4683901603659359 - 6.3956369988602974e-19j,
                                0.8835217358233883 - 8.1864153585411806e-17j,
                            ],
                            [
                                -0.8329736786855172 + 2.9455985518989175e-01j,
                                -0.4415926163678483 + 1.5615794407274164e-01j,
                            ],
                        ]
                    ]
                ),
                np.array(
                    [
                        [
                            [
                                -0.4058196959724421 + 0.5151423247988386j,
                                -0.4840321924212571 - 0.5761146990280315j,
                            ],
                            [
                                -0.5909518767589903 + 0.4689836321729041j,
                                0.2954649313420682 + 0.5828173082896994j,
                            ],
                        ],
                        [
                            [
                                -0.0260113148536681 - 0.0094803323978315j,
                                -0.0540176442214368 - 0.0130506905522443j,
                            ],
                            [
                                -0.0011067905722662 - 0.0011408253412135j,
                                -0.0359893658360522 - 0.0494539068657245j,
                            ],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [-0.6311207544789622 + 0.0j],
                            [-0.6489048744205389 + 0.2205422674772881j],
                        ],
                        [
                            [0.0502563402775399 + 0.0j],
                            [-0.0438176022650823 + 0.0148922187825892j],
                        ],
                    ]
                ),
            ]
        ),
        np.asarray(
            [
                np.array(
                    [
                        [
                            [
                                -0.7070516516575543 - 1.5756236002135627e-18j,
                                0.7071619064177063 - 9.4537416012813769e-18j,
                            ],
                            [
                                -0.7064056577579271 + 3.2695696596265186e-02j,
                                -0.7062955209057308 + 3.2690598957162323e-02j,
                            ],
                        ]
                    ]
                ),
                np.array(
                    [
                        [
                            [
                                -0.461002318087147 - 0.1976981512184319j,
                                0.083993534209597 + 0.6984803516638729j,
                            ],
                            [
                                -0.8026595817951936 - 0.1262428784845123j,
                                -0.1324883193859068 - 0.5315515337970971j,
                            ],
                        ],
                        [
                            [
                                -0.1854038695512307 + 0.0679702227545758j,
                                -0.0068962187315163 + 0.0256522839117862j,
                            ],
                            [
                                -0.0024458528228437 + 0.2217929369603371j,
                                0.45190934140008 + 0.0065150861395994j,
                            ],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [
                                -0.652061930495123 + 1.5097515251860649e-17j,
                                0.5559871096662411 - 4.7461909978956940e-01j,
                            ],
                            [
                                -0.7162693709194035 + 2.3652144533576384e-03j,
                                -0.4759855049610297 + 3.9939865972295069e-01j,
                            ],
                        ],
                        [
                            [
                                0.1615415772348854 + 7.0473141211557788e-19j,
                                0.2014472273728642 - 1.8491737919965054e-01j,
                            ],
                            [
                                0.1886991910684929 + 8.0477920479822464e-03j,
                                -0.0640618495706462 + 2.5577540181354282e-02j,
                            ],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [0.6386421750678314 + 0.0j],
                            [0.4677619923909142 - 0.5869869259512422j],
                        ],
                        [
                            [0.0110400534559703 + 0.0j],
                            [-0.0058542545334448 + 0.0073464089178308j],
                        ],
                    ]
                ),
            ]
        ),
        np.asarray(
            [
                np.array(
                    [
                        [
                            [
                                -0.7572758074506795 + 7.6825084630161755e-19j,
                                0.6530952085645111 + 4.9168054163303523e-17j,
                            ],
                            [
                                -0.5841994111038741 + 2.9196643559801294e-01j,
                                -0.6773898735657294 + 3.3854040784031025e-01j,
                            ],
                        ]
                    ]
                ),
                np.array(
                    [
                        [
                            [
                                -0.3894661746850117 - 0.747117487647773j,
                                0.4088359109211077 + 0.3168487374784127j,
                            ],
                            [
                                -0.1598783575995494 - 0.5011716221550987j,
                                -0.5520699581760301 - 0.6255371583076383j,
                            ],
                        ],
                        [
                            [
                                0.0452666167611235 + 0.0552899490004448j,
                                0.1426057935313436 - 0.0417789824127418j,
                            ],
                            [
                                -0.076360451767216 + 0.0496035351284366j,
                                0.0653955476655202 - 0.1001180652077502j,
                            ],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [
                                -0.6264963118736763 - 0.4187924103538984j,
                                -0.0456666510668808 + 0.5783243828381792j,
                            ],
                            [
                                -0.6188557746877741 - 0.1432175581446703j,
                                -0.1887153991701832 - 0.7250101988023703j,
                            ],
                        ],
                        [
                            [
                                -0.1216417309744934 - 0.0800774942917764j,
                                -0.0837699339972569 + 0.0838239421336974j,
                            ],
                            [
                                -0.0788089334646137 - 0.0346638949481445j,
                                0.0027874289783829 + 0.2969015831327159j,
                            ],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [
                                -0.7068490809473519 + 9.9746599868666408e-18j,
                                -0.2630470177245881 - 6.5546606916509087e-01j,
                            ],
                            [
                                -0.6784663512400824 + 1.9609492896560626e-01j,
                                0.4351317858877239 + 5.5727943014781700e-01j,
                            ],
                        ],
                        [
                            [
                                0.0155911994726705 - 4.3368086899420177e-19j,
                                -0.0311799274051314 + 1.0718271280453315e-02j,
                            ],
                            [
                                -0.0320136720787598 - 1.8072219893071277e-02j,
                                -0.0125116955417412 - 5.1819442479402681e-03j,
                            ],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [-0.7448053658792243 + 0.0j],
                            [-0.577273769629304 - 0.0555787948553156j],
                        ],
                        [
                            [0.0344264867746061 + 0.0j],
                            [-0.0440095111201809 - 0.0042371500645213j],
                        ],
                    ]
                ),
            ]
        ),
        np.asarray(
            [
                np.array(
                    [
                        [
                            [
                                -0.7430203937135357 + 0.0000000000000000e00j,
                                0.6692687759979412 + 7.2790351442815188e-17j,
                            ],
                            [
                                -0.6288920930928306 - 2.2894416299853779e-01j,
                                -0.6981943090298925 - 2.5417319353639284e-01j,
                            ],
                        ]
                    ]
                ),
                np.array(
                    [
                        [
                            [
                                -0.2483965927209076 + 0.407003156136946j,
                                0.7994703943382407 + 0.3463759598213064j,
                            ],
                            [
                                -0.8743089105877639 + 0.045056342654096j,
                                -0.0833409559465735 - 0.4716667158567753j,
                            ],
                        ],
                        [
                            [
                                -0.0149713634129998 - 0.0499523354620145j,
                                -0.0371315118172701 - 0.0063657173892792j,
                            ],
                            [
                                -0.0547630028316919 - 0.021978499789232j,
                                -0.0991833526330835 + 0.0141056117225484j,
                            ],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [
                                -0.6437464658906025 - 0.3530982087362634j,
                                0.0937510967666496 - 0.0360404956200982j,
                            ],
                            [
                                -0.4114172670722651 - 0.1828452270256108j,
                                -0.1184510675380616 + 0.719154715448606j,
                            ],
                        ],
                        [
                            [
                                0.3715602819473786 + 0.2081908404626114j,
                                0.0203447618238633 + 0.2903014237802751j,
                            ],
                            [
                                0.2419830345833367 + 0.1351271244045365j,
                                -0.0806928226423744 + 0.6062157180155572j,
                            ],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [
                                -0.2596690814921865 - 0.5510199842764877j,
                                0.5501093103899354 - 0.5582931266905842j,
                            ],
                            [
                                -0.5974589050072321 - 0.5166649038713198j,
                                -0.2107688169030668 + 0.5703912809924573j,
                            ],
                        ],
                        [
                            [
                                0.0306471554522822 + 0.0548966181008789j,
                                0.0451661934397042 + 0.0183753769540044j,
                            ],
                            [
                                0.0324998335494694 - 0.0063287426215501j,
                                0.0676176251900073 + 0.09470502075949j,
                            ],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [
                                -0.8374788408497464 + 0.0000000000000000e00j,
                                -0.5245919291187179 + 6.2386148522301690e-02j,
                            ],
                            [
                                -0.5037174404758303 - 1.6516232518760160e-01j,
                                0.8297419785068189 + 1.5735399803485478e-01j,
                            ],
                        ],
                        [
                            [
                                0.044324686835429 + 2.1684043449710089e-19j,
                                -0.048801608310124 - 3.7574460321160325e-02j,
                            ],
                            [
                                -0.0019839290878767 + 1.2510285284670272e-01j,
                                0.0300492956956814 + 5.4615451734875121e-02j,
                            ],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [0.5381468033714101 + 0.0j],
                            [0.7265811026216296 - 0.1646525351657372j],
                        ],
                        [
                            [0.0656297622513345 + 0.0j],
                            [-0.0462347806123897 + 0.0104773903604068j],
                        ],
                    ]
                ),
            ]
        ),
        np.asarray(
            [
                np.array(
                    [
                        [
                            [
                                -0.5458337035453097 - 2.4263907767435086e-17j,
                                0.8378935302733943 + 0.0000000000000000e00j,
                            ],
                            [
                                -0.6234855958008547 - 5.5976001992180935e-01j,
                                -0.4061607347082539 - 3.6464762375098531e-01j,
                            ],
                        ]
                    ]
                ),
                np.array(
                    [
                        [
                            [
                                -0.6850793037227788 - 0.2072484126930899j,
                                0.6234340388881175 + 0.0342199857112904j,
                            ],
                            [
                                -0.5548626022045593 - 0.4096144965786633j,
                                -0.520088065312388 - 0.2190343336747689j,
                            ],
                        ],
                        [
                            [
                                0.0085389987888449 + 0.0903063757727762j,
                                0.0817491475678125 + 0.4369951289489011j,
                            ],
                            [
                                -0.0411450616700694 + 0.0462258348646831j,
                                -0.0409911055909914 + 0.3039135076533009j,
                            ],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [
                                -0.7071595401122749 - 0.4145153796215556j,
                                0.4942346366467875 + 0.1602125735316647j,
                            ],
                            [
                                -0.516982242548026 - 0.2242461666350387j,
                                -0.7551484456365614 - 0.0777533616595736j,
                            ],
                        ],
                        [
                            [
                                0.0619092446826147 + 0.0330391212738003j,
                                -0.0232100834111218 + 0.3173093307155614j,
                            ],
                            [
                                0.0690414535229326 + 0.0292291311643433j,
                                -0.1011094670750726 + 0.2057237258096059j,
                            ],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [
                                -0.1282553483289102 - 0.6838355547355647j,
                                0.1108754527752037 - 0.7024776874115879j,
                            ],
                            [
                                -0.0689525601670817 - 0.7142400854497385j,
                                -0.1678300288146647 + 0.6705445572317588j,
                            ],
                        ],
                        [
                            [
                                -0.0011877795235857 - 0.0256679460301157j,
                                -0.0683814301371949 - 0.044423515359312j,
                            ],
                            [
                                -0.0125958091738332 + 0.0143956870396086j,
                                -0.0736549305434746 - 0.0660325174277289j,
                            ],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [
                                -0.3025231319322409 - 0.6266777084192915j,
                                0.6272080869009208 + 0.3216395397125277j,
                            ],
                            [
                                -0.6162903312892221 - 0.3601643218454526j,
                                -0.6792670677855053 + 0.0975789728427737j,
                            ],
                        ],
                        [
                            [
                                -0.0245398333106586 - 0.0669756790252716j,
                                0.0252600990024095 - 0.1175434639128661j,
                            ],
                            [
                                -0.0281244275746179 - 0.0185408958540085j,
                                0.05852327349588 - 0.1198044441985012j,
                            ],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [
                                -0.6081181385664572 - 2.0762471603097410e-17j,
                                -0.6514666461461877 + 3.3023354658959325e-01j,
                            ],
                            [
                                -0.6951681374653502 + 2.9162596095936338e-01j,
                                0.4036011770940315 - 3.2214231802227389e-01j,
                            ],
                        ],
                        [
                            [
                                0.1441167065276071 + 6.7762635780344027e-19j,
                                -0.1928554256551324 + 3.2720503668196349e-01j,
                            ],
                            [
                                0.1696254645766093 - 1.1111015147481074e-01j,
                                0.1534787992657325 + 1.7904004410948635e-01j,
                            ],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [-0.7541597571190264 + 0.0j],
                            [-0.6109485993144637 - 0.0187863039697647j],
                        ],
                        [
                            [-0.0184132282467146 + 0.0j],
                            [0.0227079628356306 + 0.000698256273675j],
                        ],
                    ]
                ),
            ]
        ),
        np.asarray(
            [
                np.array(
                    [
                        [
                            [
                                -0.4598370633633478 + 5.4517441427012262e-18j,
                                0.8880033080779444 + 2.1806976570804905e-17j,
                            ],
                            [
                                -0.870272763063613 + 1.7656498244838031e-01j,
                                -0.4506556091085561 + 9.1431104234960592e-02j,
                            ],
                        ]
                    ]
                ),
                np.array(
                    [
                        [
                            [
                                -0.7981544848277027 + 0.1160218107787375j,
                                0.4416248001277366 + 0.2621619562098562j,
                            ],
                            [
                                -0.4916287640600067 + 0.2585721411999246j,
                                -0.7906855671753772 - 0.1576340858937572j,
                            ],
                        ],
                        [
                            [
                                -0.1632310872787213 + 0.0896685390122318j,
                                0.076320883815995 - 0.0299953816423727j,
                            ],
                            [
                                -0.0106856284381674 + 0.0783003609140051j,
                                0.2630718173974106 - 0.1013661290483491j,
                            ],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [
                                -0.4118982430329323 - 0.3455216186103894j,
                                0.6725108493669645 - 0.4827605957206877j,
                            ],
                            [
                                -0.7071506928437178 - 0.4508441300716164j,
                                -0.4002606680098351 + 0.3542137103334215j,
                            ],
                        ],
                        [
                            [
                                0.0404321921956464 + 0.0254975530952142j,
                                -0.0202758496278627 - 0.0911346014916821j,
                            ],
                            [
                                0.0692608147229438 + 0.0234556733900642j,
                                -0.1422575386224151 - 0.0064480072266461j,
                            ],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [
                                -0.6328933610775296 - 0.5326163524450195j,
                                0.5237737560581593 + 0.1875126520637548j,
                            ],
                            [
                                -0.4056541243036277 - 0.3854756741618124j,
                                -0.751745492756337 - 0.3124060213112616j,
                            ],
                        ],
                        [
                            [
                                0.0334055426600222 + 0.0321987345490581j,
                                0.0853934724636376 + 0.1212010482940789j,
                            ],
                            [
                                0.0187908744958871 - 0.0106425995796298j,
                                0.0024785148552132 + 0.0761141738894634j,
                            ],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [
                                -0.6353123100331366 - 0.4593511920588956j,
                                0.4770685112393402 + 0.377140817545966j,
                            ],
                            [
                                -0.2949165380069693 - 0.5427122812993408j,
                                -0.3336906032846231 - 0.7063795360320165j,
                            ],
                        ],
                        [
                            [
                                -0.018967585106355 - 0.024450306416635j,
                                -0.0231440981781905 + 0.0816878916515189j,
                            ],
                            [
                                -0.0375519712595642 - 0.038660433225283j,
                                -0.0116466156948684 + 0.1118247159588658j,
                            ],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [
                                -0.4326793418278404 - 0.631186598475257j,
                                0.3427698546363083 - 0.4905273925036116j,
                            ],
                            [
                                -0.4565747001263349 - 0.4284436325516723j,
                                -0.291301666653459 + 0.7180582571222054j,
                            ],
                        ],
                        [
                            [
                                -0.0518356016190636 - 0.0633884758957934j,
                                0.0148169649901143 - 0.0242219815641124j,
                            ],
                            [
                                -0.0698243812324207 - 0.1038613946808124j,
                                0.0889403263109822 - 0.180861145227589j,
                            ],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [
                                -0.6699467464751714 - 3.9031278209478160e-18j,
                                -0.6977182319774758 - 9.6839188029619550e-02j,
                            ],
                            [
                                -0.6950809476619499 - 1.6183723444869344e-01j,
                                0.5762213815238573 + 2.6933762753530061e-01j,
                            ],
                        ],
                        [
                            [
                                0.1141851986394764 + 4.3368086899420177e-19j,
                                -0.2693013339847729 + 2.3149072064395759e-02j,
                            ],
                            [
                                0.1238725483128291 + 1.1601670059455256e-01j,
                                -0.0745484372794617 + 1.4360029516735601e-01j,
                            ],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [0.6688304751895966 + 0.0j],
                            [0.554483402272025 - 0.0061797942804347j],
                        ],
                        [
                            [-0.0837347769338295 + 0.0j],
                            [0.100990245601556 - 0.0011255502682153j],
                        ],
                    ]
                ),
            ]
        ),
        np.asarray(
            [
                np.array(
                    [
                        [
                            [
                                -0.8607440533600228 + 1.0097322848450451e-19j,
                                0.5090379893537986 + 5.1698292984066308e-17j,
                            ],
                            [
                                -0.4410087608424453 + 2.5422617384047785e-01j,
                                -0.7457118649566556 + 4.2987688918748423e-01j,
                            ],
                        ]
                    ]
                ),
                np.array(
                    [
                        [
                            [
                                -7.5646996764106167e-01 - 0.0719466977157099j,
                                -3.8243150634142742e-02 - 0.6401961534705833j,
                            ],
                            [
                                -6.2561162781208268e-01 - 0.1543102207338461j,
                                -6.0992456138451667e-02 + 0.7614911751317406j,
                            ],
                        ],
                        [
                            [
                                -2.2459141680528736e-02 - 0.0190859353845969j,
                                5.9385434141056952e-04 - 0.0184531508571932j,
                            ],
                            [
                                6.6832510193897562e-02 - 0.0451667718015712j,
                                6.7985249909101986e-02 + 0.0116013515153914j,
                            ],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [
                                -0.5732903945285319 - 0.2816809826500967j,
                                0.2591937524733295 + 0.592177965719108j,
                            ],
                            [
                                -0.6815761166448715 - 0.0953400005661647j,
                                -0.4922548487697549 - 0.2543768580693652j,
                            ],
                        ],
                        [
                            [
                                0.2408031380015913 + 0.1202850395789868j,
                                -0.0956329554123388 + 0.371135494494846j,
                            ],
                            [
                                0.2127301975485183 + 0.025484310839671j,
                                -0.3274931935181032 + 0.1448531128470462j,
                            ],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [
                                -0.5703753333978321 - 0.4105369990204051j,
                                0.5680720955638067 + 0.4254232380521883j,
                            ],
                            [
                                -0.6001343039919134 - 0.3804238102589699j,
                                -0.586732177709093 - 0.3880779539009969j,
                            ],
                        ],
                        [
                            [
                                -0.0080447747226261 - 0.003372971985387j,
                                -0.0056040350647109 - 0.0122431428804646j,
                            ],
                            [
                                0.0337792739665356 + 0.0055460397368544j,
                                -0.0322065658716398 - 0.015213594639508j,
                            ],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [
                                -0.6209946294001301 - 0.4921524066327827j,
                                0.1959902936013406 + 0.3004131234815642j,
                            ],
                            [
                                -0.4041988495717739 - 0.3344311200339942j,
                                -0.600876026670115 - 0.4693209463375816j,
                            ],
                        ],
                        [
                            [
                                -0.1539386050422358 + 0.0173030564534223j,
                                0.2530366174709696 - 0.2354590117057856j,
                            ],
                            [
                                -0.2407366415942101 - 0.1223935225752507j,
                                0.4045724919264119 - 0.0829351117312868j,
                            ],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [
                                -0.5488099109698334 - 0.5039474185111208j,
                                0.1838080950162287 - 0.5763324037382751j,
                            ],
                            [
                                -0.4579674730481228 - 0.3888436026326899j,
                                -0.2578009269539564 + 0.6624405326087648j,
                            ],
                        ],
                        [
                            [
                                -0.1840013276371877 - 0.1431340455801799j,
                                0.0522009617344365 + 0.2769636892407512j,
                            ],
                            [
                                -0.0843950451598841 - 0.1498157538274002j,
                                0.1957481345493265 - 0.1049552746796238j,
                            ],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [
                                -0.1767213996342019 - 0.6493356209922121j,
                                0.6053161698801638 + 0.3974311110766725j,
                            ],
                            [
                                -0.3126470599068834 - 0.6680814020429926j,
                                -0.5900908333443157 - 0.2694025763928804j,
                            ],
                        ],
                        [
                            [
                                0.0169849796475122 + 0.0387068832174426j,
                                0.1762497150921175 - 0.0477853743278467j,
                            ],
                            [
                                0.0219724607401001 + 0.0279705742670205j,
                                0.1045701746991879 - 0.1028279201361973j,
                            ],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [
                                -0.6074464215003265 - 5.7245874707234634e-17j,
                                -0.7652362312931738 + 2.0634742397066397e-01j,
                            ],
                            [
                                -0.721244727530714 + 3.2981027051300460e-01j,
                                0.4625279998909902 - 3.8509399771987118e-01j,
                            ],
                        ],
                        [
                            [
                                0.0384636188377573 + 3.2526065174565133e-19j,
                                -0.0838676905556021 + 1.6983851654629394e-02j,
                            ],
                            [
                                0.0168629313109932 - 1.6621222976583135e-02j,
                                -0.0454736414390205 + 1.4649927371079924e-02j,
                            ],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [-0.5783006773729699 + 0.0j],
                            [-0.7349956609263215 + 0.2650483613285932j],
                        ],
                        [
                            [0.0224575761701841 + 0.0j],
                            [-0.0156364284089663 + 0.0056386859775541j],
                        ],
                    ]
                ),
            ]
        ),
        np.asarray(
            [
                np.array(
                    [
                        [
                            [
                                -0.6584394535047408 + 9.040012560365488e-20j,
                                0.7526336997958419 + 5.785608038633912e-18j,
                            ],
                            [
                                -0.7514349597711147 + 4.246159796994165e-02j,
                                -0.6573907392005717 + 3.714740832074133e-02j,
                            ],
                        ]
                    ]
                ),
                np.array(
                    [
                        [
                            [
                                -0.537529759720264 + 0.5773586524511798j,
                                0.5957033839075052 - 0.0788459623262125j,
                            ],
                            [
                                -0.2966375224224164 + 0.5336325134589134j,
                                -0.7361122077723212 + 0.2843642748417394j,
                            ],
                        ],
                        [
                            [
                                0.0150843460251642 - 0.0471834288650409j,
                                -0.0450994460320081 + 0.0055248661558608j,
                            ],
                            [
                                -0.0065963860464827 - 0.0496380797584114j,
                                -0.0872428532448384 + 0.0807512833802459j,
                            ],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [
                                -0.3217914541545513 - 0.7248292426931507j,
                                0.3315967771941917 + 0.5067868015892226j,
                            ],
                            [
                                -0.4013144745153958 - 0.4565809250419096j,
                                -0.6091016482117848 - 0.5040265307168429j,
                            ],
                        ],
                        [
                            [
                                0.0187755322293345 + 0.0303596440667982j,
                                0.0560412216507494 - 0.0319083498330856j,
                            ],
                            [
                                -0.0154368926761584 + 0.0063917249961212j,
                                0.0143019722575102 - 0.0616428896547561j,
                            ],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [
                                -0.5665060388607093 - 0.7222096703556217j,
                                -0.0467137999548778 + 0.3618160643419908j,
                            ],
                            [
                                -0.3530099281790957 - 0.1017073587217815j,
                                -0.4626390734557525 - 0.8043077619792502j,
                            ],
                        ],
                        [
                            [
                                0.0476430770003577 + 0.0028041733077835j,
                                -0.0254550154459602 + 0.0341764763456488j,
                            ],
                            [
                                0.0909803339196563 + 0.1094008977745023j,
                                -0.0418844380809142 - 0.0488955285242689j,
                            ],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [
                                -0.4265270130604106 - 0.5731771675242459j,
                                0.3322034622320452 - 0.3621191044315397j,
                            ],
                            [
                                -0.4189984557871408 - 0.2012794803207656j,
                                -0.319288220155853 + 0.5551848999341619j,
                            ],
                        ],
                        [
                            [
                                -0.3658204351438121 - 0.262499889072301j,
                                -0.0340026677774641 + 0.4121464063374005j,
                            ],
                            [
                                -0.1665172974490576 - 0.2073902450138987j,
                                0.3090574822191386 - 0.2860035039175093j,
                            ],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [
                                -0.4883807447550823 - 0.5416737751492168j,
                                0.586706692661832 - 0.1411731180230433j,
                            ],
                            [
                                -0.4896045731132726 - 0.4270587403572575j,
                                -0.7047998857982777 + 0.239457457923614j,
                            ],
                        ],
                        [
                            [
                                0.0981019982138236 + 0.1085496973940759j,
                                -0.0090544706837816 - 0.0483365161889412j,
                            ],
                            [
                                0.1138961359809312 + 0.10771534287107j,
                                -0.2777843958525662 + 0.0466913433212099j,
                            ],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [
                                -0.2140080360697484 - 5.0205616462730362e-01j,
                                0.8341253530662697 - 7.7344607475358484e-02j,
                            ],
                            [
                                -0.0310783436760627 - 8.3717910228567638e-01j,
                                -0.5253947971435565 - 1.4731547544149071e-01j,
                            ],
                        ],
                        [
                            [
                                0.0089126412247618 + 1.2335660054756722e-02j,
                                0.0138485103099943 + 6.0013334869472943e-04j,
                            ],
                            [
                                -0.00434603653136 - 7.4137434317828209e-03j,
                                0.017198766378642 + 4.8193291745362308e-03j,
                            ],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [
                                -0.5141324415881146 - 0.5857754414607566j,
                                0.2988158204840621 + 0.5361106718284355j,
                            ],
                            [
                                -0.5388760925591876 - 0.2710340571958791j,
                                -0.6192211259780797 - 0.4643498143314865j,
                            ],
                        ],
                        [
                            [
                                0.0941305730654242 + 0.1206613318760576j,
                                -0.0123322930683984 + 0.1163002000547795j,
                            ],
                            [
                                -0.0012936030021226 + 0.0725718002513804j,
                                -0.0781875613355543 - 0.0666917035330909j,
                            ],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [
                                -0.6164868659184163 + 3.2959746043559335e-17j,
                                -0.5542499618338282 - 3.4103069190184176e-01j,
                            ],
                            [
                                -0.689164541285134 - 3.0618789234704435e-01j,
                                0.3920331136530223 + 4.7662365245917110e-01j,
                            ],
                        ],
                        [
                            [
                                0.1564728574775188 - 4.5536491244391186e-18j,
                                0.066713855629527 - 1.4665144960478310e-01j,
                            ],
                            [
                                0.1604173088199458 - 3.2057536854853429e-02j,
                                0.408875141865695 + 5.0087032895884065e-02j,
                            ],
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [0.8308581261935942 + 0.0j],
                            [0.4311566750490988 - 0.2601220554772691j],
                        ],
                        [
                            [0.0147521766994044 + 0.0j],
                            [-0.0208419379402228 + 0.0125741941407333j],
                        ],
                    ]
                ),
            ]
        ),
    ],
)
def test_get_parameters_for_MPS_layer_is_correct_size(mps):
    number_of_qubits = len(mps)
    expected_number_of_parameters = 15 * (number_of_qubits - 1) + 3
    parameters = get_parameters_for_MPS_layer(mps)
    assert len(parameters) == expected_number_of_parameters


def test_get_parameters_for_MPS_layer_initializes_in_range_0_to_2pi():
    mps = np.asarray(
        [
            np.array(
                [
                    [
                        [
                            -0.7430203937135357 + 0.0000000000000000e00j,
                            0.6692687759979412 + 7.2790351442815188e-17j,
                        ],
                        [
                            -0.6288920930928306 - 2.2894416299853779e-01j,
                            -0.6981943090298925 - 2.5417319353639284e-01j,
                        ],
                    ]
                ]
            ),
            np.array(
                [
                    [
                        [
                            -0.2483965927209076 + 0.407003156136946j,
                            0.7994703943382407 + 0.3463759598213064j,
                        ],
                        [
                            -0.8743089105877639 + 0.045056342654096j,
                            -0.0833409559465735 - 0.4716667158567753j,
                        ],
                    ],
                    [
                        [
                            -0.0149713634129998 - 0.0499523354620145j,
                            -0.0371315118172701 - 0.0063657173892792j,
                        ],
                        [
                            -0.0547630028316919 - 0.021978499789232j,
                            -0.0991833526330835 + 0.0141056117225484j,
                        ],
                    ],
                ]
            ),
            np.array(
                [
                    [
                        [
                            -0.6437464658906025 - 0.3530982087362634j,
                            0.0937510967666496 - 0.0360404956200982j,
                        ],
                        [
                            -0.4114172670722651 - 0.1828452270256108j,
                            -0.1184510675380616 + 0.719154715448606j,
                        ],
                    ],
                    [
                        [
                            0.3715602819473786 + 0.2081908404626114j,
                            0.0203447618238633 + 0.2903014237802751j,
                        ],
                        [
                            0.2419830345833367 + 0.1351271244045365j,
                            -0.0806928226423744 + 0.6062157180155572j,
                        ],
                    ],
                ]
            ),
            np.array(
                [
                    [
                        [
                            -0.2596690814921865 - 0.5510199842764877j,
                            0.5501093103899354 - 0.5582931266905842j,
                        ],
                        [
                            -0.5974589050072321 - 0.5166649038713198j,
                            -0.2107688169030668 + 0.5703912809924573j,
                        ],
                    ],
                    [
                        [
                            0.0306471554522822 + 0.0548966181008789j,
                            0.0451661934397042 + 0.0183753769540044j,
                        ],
                        [
                            0.0324998335494694 - 0.0063287426215501j,
                            0.0676176251900073 + 0.09470502075949j,
                        ],
                    ],
                ]
            ),
            np.array(
                [
                    [
                        [
                            -0.8374788408497464 + 0.0000000000000000e00j,
                            -0.5245919291187179 + 6.2386148522301690e-02j,
                        ],
                        [
                            -0.5037174404758303 - 1.6516232518760160e-01j,
                            0.8297419785068189 + 1.5735399803485478e-01j,
                        ],
                    ],
                    [
                        [
                            0.044324686835429 + 2.1684043449710089e-19j,
                            -0.048801608310124 - 3.7574460321160325e-02j,
                        ],
                        [
                            -0.0019839290878767 + 1.2510285284670272e-01j,
                            0.0300492956956814 + 5.4615451734875121e-02j,
                        ],
                    ],
                ]
            ),
            np.array(
                [
                    [
                        [0.5381468033714101 + 0.0j],
                        [0.7265811026216296 - 0.1646525351657372j],
                    ],
                    [
                        [0.0656297622513345 + 0.0j],
                        [-0.0462347806123897 + 0.0104773903604068j],
                    ],
                ]
            ),
        ]
    )
    parameters = get_parameters_for_MPS_layer(mps)
    for parameter in parameters:
        assert parameter >= 0 * np.pi
        assert parameter <= 2 * np.pi


def test_get_parameters_for_MPS_layer_first_tensor_explicitly_for_6_qubit_mps():
    mps = np.asarray(
        [
            np.array(
                [
                    [
                        [-0.63442861 - 3.55507664e-19j, 0.77298146 - 4.55049809e-17j],
                        [-0.60854993 + 4.76620731e-01j, -0.49947056 + 3.91188977e-01j],
                    ]
                ]
            ),
            np.array(
                [
                    [
                        [-0.92222366 - 0.1819367j, 0.30941761 - 0.12475666j],
                        [-0.24489206 - 0.2297901j, -0.92632779 - 0.15336545j],
                    ],
                    [
                        [0.02129682 + 0.03788334j, -0.06106673 + 0.00155094j],
                        [-0.03897827 - 0.01479676j, -0.02676018 - 0.05142686j],
                    ],
                ]
            ),
            np.array(
                [
                    [
                        [-0.46924002 - 0.31978182j, 0.57373457 - 0.05723614j],
                        [-0.5975817 - 0.39090355j, -0.42626552 - 0.27652741j],
                    ],
                    [
                        [-0.21104345 - 0.19910946j, -0.26175411 + 0.2814635j],
                        [-0.24879787 - 0.14683198j, 0.15981032 + 0.48590813j],
                    ],
                ]
            ),
            np.array(
                [
                    [
                        [-0.40667933 - 0.56160256j, 0.68125907 - 0.21949277j],
                        [-0.20389561 - 0.686802j, -0.69309907 - 0.02271811j],
                    ],
                    [
                        [-0.02488891 - 0.02792529j, -0.03972115 + 0.0533864j],
                        [-0.0395187 - 0.05461744j, 0.04655174 + 0.01454731j],
                    ],
                ]
            ),
            np.array(
                [
                    [
                        [-0.86707015 + 1.51788304e-17j, -0.13996016 - 4.70448442e-01j],
                        [-0.48614104 - 9.17792293e-02j, 0.07665117 + 8.52276039e-01j],
                    ],
                    [
                        [0.05371716 + 1.51788304e-18j, -0.13792981 - 3.29157650e-02j],
                        [-0.01614852 + 1.69266756e-02j, -0.08089098 + 1.37163838e-02j],
                    ],
                ]
            ),
            np.array(
                [
                    [[0.68074106 + 0.0j], [0.65144518 - 0.28898916j]],
                    [[-0.01045085 + 0.0j], [0.00912508 - 0.004048j]],
                ]
            ),
        ]
    )
    parameters = get_parameters_for_MPS_layer(mps)
    first_tensor_circuit = cirq.Circuit()
    qubits = [cirq.LineQubit(0)]
    first_tensor_circuit.append(cirq.rz(parameters[-3]).on(qubits[0]))
    first_tensor_circuit.append(cirq.ry(parameters[-2]).on(qubits[0]))
    first_tensor_circuit.append(cirq.rz(parameters[-1]).on(qubits[0]))
    unitary = cirq.unitary(first_tensor_circuit)
    phase = unitary[0][0] / mps[0].reshape(2, 2)[0][0]
    np.testing.assert_array_almost_equal(unitary, phase * mps[0].reshape(2, 2), 6)


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
    assert overlap > (1 - 1e-14)


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

    assert overlap > (1 - 1e-14)


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
